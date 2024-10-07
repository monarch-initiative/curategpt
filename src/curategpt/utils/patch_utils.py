from typing import Dict, List


def patches_to_oak_commands(patch_dict: Dict[str, List[Dict]], ont_path: str) -> str:
    for id, patches in patch_dict.items():
        notes = ""
        changes = []
        for patch in patches:
            path = patch["path"]
            op = patch["op"]
            v = patch["value"]
            if path == "/review_notes":
                notes = patch["value"]
                if isinstance(notes, list):
                    notes = ". ".join(notes)
                notes = notes.replace('"', "'")
            elif path == "definition":
                if op == "replace":
                    changes.append(f"change definition of {id} to {v}")
        if changes:
            for change in changes:
                # print(f'runoak -i {ont_path} apply "{change}"')
                print(f'robot kgcl:apply -i {ont_path} apply -k "{change}"')
            if not notes:
                notes = f"Applying {len(changes)}"
            print(f'git commit -m "{notes}"')
