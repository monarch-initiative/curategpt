"""
Formats data objects for presentation to humans and machine agents.

Note: this package is not in heavy use just now.

Currently we remove identifiers from data objects before caching,
and present the LLM with the same kind of view a human would see
when using a curation tool like Protege.

In future, this will be separated, and the store will house the
normalized representation, and this layer will handle
presenting data to the LLM (or human) with IDs changed to labels.

TODO: will this also handle the reverse?
"""
