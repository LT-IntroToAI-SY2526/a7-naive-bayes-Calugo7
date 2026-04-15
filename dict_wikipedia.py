import wikipedia 
import re
from typing import List

article = wikipedia.page("Artemis II", auto_suggest=False).content

print(article)