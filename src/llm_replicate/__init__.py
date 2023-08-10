import json

import click
import replicate
import requests
import sqlite_utils

import llm


@llm.hookimpl
def register_commands(cli):
    @cli.group(name="replicate")
    def replicate():
        "Commands for working with models hosted on Replicate"

    @replicate.command(name="fetch-models")
    @click.option("--key", "-k", help="Replicate API key")
    def fetch_models(key):
        """
        Fetch details of models in the language-models collection

        https://replicate.com/collections/language-models
        """
        token = llm.get_key(key, "replicate", env_var="REPLICATE_API_TOKEN")
        if not token:
            raise click.ClickException(
                "Pass --key, store a 'replicate' key or set the REPLICATE_API_TOKEN environment variable."
            )
        response = requests.get(
            "https://api.replicate.com/v1/collections/language-models",
            headers={"Authorization": "Token {}".format(token)},
        )
        if response.status_code != 200:
            raise click.ClickException("Error fetching models: {}".format(response.text))
        models = response.json()["models"]
        json_path = config_dir() / "fetch-models.json"
        with open(json_path, "w") as fp:
            json.dump(models, fp, indent=2)

    @replicate.command(name="add")
    @click.argument("model_id")
    @click.option("--chat", is_flag=True, help="This is a chat model")
    @click.option("aliases", "--alias", multiple=True, help="Aliases for this model")
    @click.option("--version", help="Model version (defaults to latest)")
    @click.option("--key", "-k", help="Replicate API key")
    def add_model(model_id, chat, aliases, version, key):
        """
        Register additional Replicate models with LLM

        Example usage:

            llm replicate add joehoover/falcon-40b-instruct

        \b
        Use --chat for "chat" models that should be prompted using
        'User: ... \\nAssistant:' format.
        """
        if not version:
            # Fetch latest version from Replicate API
            token = llm.get_key(key, "replicate", env_var="REPLICATE_API_TOKEN")
            if not token:
                raise click.ClickException(
                    "Pass --key, store a 'replicate' key or set the REPLICATE_API_TOKEN environment variable."
                )
            response = requests.get(
                "https://api.replicate.com/v1/models/{}".format(model_id),
                headers={"Authorization": "Token {}".format(token)},
            )
            if response.status_code != 200:
                raise click.ClickException("Error fetching model details: {}".format(response.text))
            model_details = response.json()
            version = model_details["latest_version"]["id"]
        # Add to models.json
        models_path = config_dir() / "models.json"
        if models_path.exists():
            models = json.loads(models_path.read_text())
        else:
            models = []
        new_model = {
            "model": model_id,
            "model_id": model_id.replace("/", "-"),
            "version": version,
            "aliases": aliases,
        }
        if chat:
            new_model["chat"] = True
        updated_models = [model for model in models if model["model"] != model_id]
        updated_models.append(new_model)
        models_path.write_text(json.dumps(updated_models, indent=2))

    @replicate.command(name="edit-models")
    def edit_models():
        """
        Edit registered models using the default $EDITOR
        """
        models_path = config_dir() / "models.json"
        if not models_path.exists():
            models_path.write_text("[]")
        click.edit(filename=str(models_path))

    @replicate.command(name="fetch-predictions")
    @click.option("--key", "-k", help="Replicate API key")
    def fetch_predictions(key):
        """
        Fetch data on all Replicate predictions, save to SQLite replicate_predictions

        Example usage:

            llm replicate fetch-predictions
        """
        token = llm.get_key(key, "replicate", env_var="REPLICATE_API_TOKEN")
        db = sqlite_utils.Database(llm.user_dir() / "logs.db")
        table = db["replicate_predictions"]

        if not table.exists():

            def id_exists(id):
                return False

        else:

            def id_exists(id):
                try:
                    row = table.get(id)
                    if row["completed_at"]:
                        return True
                    # No completed_at, check if it's still running
                    if row["status"] in ("starting", "processing"):
                        return False
                    return True
                except sqlite_utils.db.NotFoundError:
                    return False

        # Need all Replicate models to guess model name
        version_to_model = {
            ma.model.version_id: "{}/{}".format(ma.model.owner, ma.model.name)
            for ma in llm.get_models_with_aliases()
            if isinstance(ma.model, ReplicateModel)
        }

        # First we fetch a list of all predictions to fetch - to get the total count
        # so we can show a progress bar
        next_url = "https://api.replicate.com/v1/predictions"
        to_fetch = []
        while next_url:
            response = requests.get(next_url, headers={"Authorization": "Token {}".format(token)})
            if response.status_code != 200:
                raise click.ClickException("Error fetching model details: {}".format(response.text))
            data = response.json()
            next_url = data.get("next")
            # For each one check if we already have it
            for result in data["results"]:
                id = result["id"]
                if not id_exists(id):
                    to_fetch.append(result["urls"]["get"])

        def get_prediction(url):
            r = requests.get(url, headers={"Authorization": "Token {}".format(token)})
            if r.status_code != 200:
                raise click.ClickException("Error fetching prediction details: {}".format(url))
            data = r.json()
            # Guess the model name, rewrite JSON with _model_guess after id
            info = {}
            info["id"] = data.pop("id")
            info["_model_guess"] = version_to_model.get(data["version"])
            # Copy remaining keys
            for key, value in data.items():
                info[key] = value
            return info

        # Fetch URLs, with a progress bar
        with click.progressbar(
            to_fetch, label="Fetching predictions", show_eta=True, show_pos=True
        ) as bar:
            for url in bar:
                info = get_prediction(url)
                table.insert(info, pk="id", replace=True, alter=True)


@llm.hookimpl
def register_models(register):
    # First do cached JSON collection
    fetch_models_path = config_dir() / "fetch-models.json"
    if fetch_models_path.exists():
        models = json.loads(fetch_models_path.read_text())
        for details in models:
            register(
                ReplicateModel(
                    owner=details["owner"],
                    name=details["name"],
                    version_id=details["latest_version"]["id"],
                    chat=False,
                ),
            )

    models_path = config_dir() / "models.json"
    if models_path.exists():
        more_models = json.loads(models_path.read_text())
        for info in more_models:
            aliases = info.get("aliases", [])
            register(
                ReplicateModel(
                    owner=info["model"].split("/")[0],
                    name=info["model"].split("/")[1],
                    version_id=info["version"],
                    chat=info.get("chat", False),
                ),
                aliases=aliases,
            )


class ReplicateModel(llm.Model):
    model_id = "replicate"
    needs_key = "replicate"
    key_env_var = "REPLICATE_API_TOKEN"

    def __init__(self, owner, name, version_id, chat):
        model_id = "replicate-{}-{}".format(owner, name)
        if model_id.startswith("replicate-replicate-"):
            model_id = model_id[len("replicate-") :]
        self.model_id = model_id
        self.version_id = version_id
        self.name = name
        self.owner = owner
        self.chat = chat

    def build_chat_prompt(self, prompt, conversation):
        prompt_lines = []
        if conversation is not None:
            for prev_response in conversation.responses:
                prompt_lines.extend(
                    [
                        f"User: {prev_response.prompt.prompt}\n",
                        f"Assistant: {prev_response.text()}\n",
                    ]
                )

        prompt_lines.extend(
            [
                f"User: {prompt.prompt}\n",
                f"Assistant:",
            ]
        )
        return prompt_lines

    def execute(self, prompt, stream, response, conversation):
        if conversation and not self.chat:
            raise llm.ModelError("Conversation mode is not supported")

        lines = [prompt.prompt]
        if self.chat:
            lines = self.build_chat_prompt(prompt, conversation)

        client = replicate.Client(api_token=self.get_key())
        output = client.run(
            "{owner}/{name}:{version_id}".format(
                owner=self.owner,
                name=self.name,
                version_id=self.version_id,
            ),
            input={"prompt": "".join(lines)},
        )
        response._prompt_json = {"lines": lines}
        yield from output

    def __str__(self) -> str:
        return "Replicate{}: {}".format(" (chat)" if self.chat else "", self.model_id)


def config_dir():
    dir_path = llm.user_dir() / "replicate"
    if not dir_path.exists():
        dir_path.mkdir()
    return dir_path
