# Curate-GPT: Original idea

(this was the original README)

## Background

Currently this README is sparse on background and assumes a lot of background knowledge on
the part of the reader about current ontology editing workflows, etc. Some newer approaches
that are of relevance:

- [LinkML-OWL](https://github.com/linkml/linkml-owl/)
- [SPIRES](https://arxiv.org/abs/2304.02711)
- [Development of Knowledge Bases using Prompt Engineering](https://bit.ly/copilot-curation)

## MVP

The goal is to have a quick proof of concept MVP. This should be ~1 week work for a dev familiar with
[LangChain](https://github.com/hwchase17/langchain). We will need to quickly iterate on this as the landscape is constantly changing.

For the first version it's likely the [OpenAI API](https://python.langchain.com/en/latest/modules/models/llms/integrations/openai.html) will be used (Mungall group community keys can be used),
but we are [actively exploring open models](https://github.com/monarch-initiative/ontogpt/issues/70). Latency is a short term concern but this will likely improve.

The MVP is a small lightweight [streamlit app](https://streamlit.io/) that presents a ChatGPT-like interface. The main
way for a curator to interact with the app is to type in a text box.

The response to questions/instructions may be any mix of text, images, interactive components, etc.

The streamlit app will be running in a space where it has access to one or more GitHub/GitLab repos.
It will likely be running in an [ODK](https://github.com/INCATools/ontology-development-kit/) Docker container.

The streamlit app will be a thin interface onto a LangChain app, implementing the
[ReAct pattern](https://python.langchain.com/en/latest/modules/agents/agents/examples/react.html).

On each instruction from the curator, the app will decide which of the various actions open to it to undertake;
e.g.

 - add a term or set of terms
 - review a term or set of terms
 - import from an external source (web, pubmed, excel, etc)

Context will be retained allowing a ChatGPT [conversational style](https://python.langchain.com/en/latest/use_cases/chatbots.html).

The app will keep in memory and sync to disk the current knowledge base (KB). The KB can be anything but
for now assume it is an ontology. The metamodel will be [LinkML](https://linkml.io), and the KB can be configured to be using
any number of data models (including the LinkML metamodel itself allowing editing of the templates used in the app).
The primary serialization will be YAML, but LinkML allows for ser/deser as rdf, json, tables, sql, etc.

There will be a background vector database synced the whole time (we will use an open self-hosted one like
[ChromaDB](https://github.com/chroma-core/chroma)). This is all
[managed automatically via LangChain](https://python.langchain.com/en/latest/reference/modules/vectorstores.html).

### Add similar entities operation

The basic form of prompt here will be something like:

- Add a term increased tentacle length
- Add phenotype terms for tentacle
- Add left and right forms for all parts of the hand
- Add a term with the definition "Catalysis of the reaction: L-rhamnose = L-rhamnulose"

When this operation is triggered, the app will first collect the most relevant N entries from the store based
on text embedding similarity (standard LangChain operation). These will be used to generate the [in-context](https://python.langchain.com/en/latest/modules/prompts/prompt_templates/examples/few_shot_examples.html) part
of the prompt (this is similar to how copilot works). The entries will be serialized as YAML, but we should
experiment with obo format (with a strategy to obviate ID hallucination). Compact forms will be best due to
the limited size of the prompt.

An example of how this might be presented as a [code-davinci](https://help.openai.com/en/articles/6195637-getting-started-with-codex) prompt:

```yaml
TentacleWidth:
  description: The width of a tentacle.
  quality: Width
  characteristic_of: Tentacle
TentacleStrength:
  description: The strength of a tentacle.
  quality: Strength
  characteristic_of: Tentacle
TentacleLength:
```

(the curator would not see this unless running in debug mode)

But it's likely a YAML embedding approach within a standard instruction-prompt model like gpt-3.5-turbo or gpt-4
will work better.

The MVP will focus on ontologies and in particular ontologies that have implicit of explicit 
[compositional patterns](https://jbiomedsem.biomedcentral.com/articles/10.1186/s13326-017-0126-0); 
this provides a lot more substrate for the LM to work with (analogous to coding with copilot).

However, it's expected the overall technique will work with less compositional ontologies where background
knowledge is relatively well known.

Our experience shows that similar approaches work well for bulk term addition.

The app will intercept the GPT response, parse it, display to the curator. Some kind of simple thumbs up/down
could be used to iterate through alternative responses until a suitable one is found. A simple text box
could be used to make minor edits on the response, but ideally this would be rare (iterations may be better done
using the NL prompt).

Standard OAK methods can be used for the display. This could be tabular, as obo, as a
[tree](https://incatools.github.io/ontology-access-kit/cli.html#runoak-tree),
as a [graph](https://github.com/INCATools/obographviz), etc.

### Import from literature

The add entities operation relies on either in-context examples or background knowledge that has been part
of LLM pre-training. This is expected to work better in some domains than others.

For more "de-novo" entity addition then external sources can be used.

In future versions the app will have access to all of PMC in vector form, but in the interim the app
will use GPT to generate queries (see [GeneGPT paper](https://arxiv.org/abs/2304.09667)) over PMC/PMID and other resources to retrieve relevant
entities.

Once relevant papers are selected [SPIRES](https://arxiv.org/abs/2304.02711) can be used to extract entities from the papers.

### Import from tables and semi-structured data

A common requirement is extracting some kind of table and converting it to an ontology formalism.

Prompts be like:

- extract table 2 of PMID:123456 into a list of disease-phenotype associations

Tables are often in messy formats, in PDFs, etc. The combination of [textract](https://textract.readthedocs.io/)
and [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) can be used to extract tables from PDFs.

Once in some kind of semi-structured format this is enough for GPT to generate suitable axioms using
in-context learning and/or using SPIRES.

### Make ontology suggestions

- make my text definitions more human readable
- make my text definitions conform to my logical definitions
- make my text definitions more consistent
- suggest missing logical definitions
- suggest missing axioms
- make general suggestions

### Integration of reasoning

We are [currently exploring](https://github.com/cmungall/gpt-reasoning-manuscript) how well GPT does at reasoning;
unsurprisingly it's not very good.

We would have an agent for performing reasoning using standard methods (e.g. robot by py4j or command line).

Explanations can be rewritten by the LLM to make them more understandable to mortals.

If the curator permits, results can be cached in the vector store. This will provide a bank of examples for
in-context learning for future MVPs.

### Review and edit

- show all the parts of the tentacle
- show all terms added in last week
- show all definitions for all neuron types

## MVP 2

- reactivity+carts: e.g. being able to click on elements in the payload and have them be added to a context/cart
- syncing with existing editing environments
- mining github issues
- github operations


