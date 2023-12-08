import logging
import openai

logger = logging.getLogger(__name__)


def generate_phenopacket(full_text, model="gpt-4-1106-preview", max_tokens=3000):
        messages = [
            {
                "role": "system",
                "content": """"You are system that returns Phenopackets in JSON format.           
These phenopackets should have phenotypicFeatures captured as HPO terms, for example:"
  "phenotypicFeatures": [
    {
      "type": {
        "id": "HP:0005294",
        "label": "Arterial dissection"
      }
    },
    {
      "type": {
        "id": "HP:0010648",
        "label": "Dermal translucency"
      }
    }]
and the correct disease diagnosis in as interpretation, preferably as OMIM diseases, for
example: 
  "interpretations": [
    {
      "id": "someuniqueID1234",
      "diagnosis": {
        "disease": {
          "id": "OMIM:130050",
          "label": "EHLERS-DANLOS SYNDROME, VASCULAR TYPE"
        }
      }
    }
  ],

Create a phenopacket from the following scientific article:
""" +
                           full_text}]

        response = openai.ChatCompletion.create(
            model=model,
            # functions=None,
            messages=messages,
            max_tokens=max_tokens,
        )
        return response.choices[0].message['content']
