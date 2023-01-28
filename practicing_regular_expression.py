# -*- coding: utf-8 -*-
"""Practicing Regular Expression.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wBYt17_FsqoP6mbfITTyQKP1VlwoLQjn
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
# %config Completer.use_jedi = False
from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

import re

pattern='[0-9]+'

df=pd.read_csv(r'C:\Users\manuj\Downloads\Input for User Attrition Prediction (1).csv', parse_dates=['year_month_dateformat'])    # importing the dataset

"""## Here begins practice for regular expressions

How to find occurance of patterns
1. First occurance
2. All occurance
"""

df['ac_created'].apply(lambda x: re.search(pattern,x))

"""search is for first occurance, I am not sure what is difference between search and find

So if you need to match at the beginning of the string, or to match the entire string use match. It is faster. Otherwise use search.

Very intrigued by the match object, what can span and match be used
"""

df['ac_created'].apply(lambda x: re.findall(pattern,x))

#lets learn the role of +, it looks one or more
pattern='\w+'
re.search(pattern,'a9888232')

"""#What is the role of bracket?

Only parentheses can be used for grouping. Square brackets define a character class, and curly braces are used by a quantifier with specific limits.

Now I would explain you what that means?
But to do that we need to know what to do with search output?

#### What are the types of regular expressions?

Metacharacters
Literals- Anything not a metacharacter is a literal. Now what is Metacharacter?

As regex is its own language, Metacharacter is grammar and literal are words of language

Metacharacter are of three type

1. Metacharacter classes
2. Quantifiers
3. Position metacharacters

#### All metacharacters start with \ to distinguish them from literals

\w = word
\W = non word
\d = Digit
\D = non digit
\s = whitespace
\S = non whitespace
"""



"""## Interestingly use \ to remove the metachacter and make them literal"""

df.columns=[re.sub("\(\)","",col) for col in df.columns]                        #This removes the metacharacter ability of (

"""I like to think of Regex is giving special powers to normal characters
e.g. w is w but in world of regex \w means word, \W means non word

#### Qunatifiers give a count

.=1
?= zero or one
+= 1 or more
* = zero or more
{min,max}

#### Position Metacharacter give a position

^ = start of the line
$= end of the line
\<=start of the word
\>=end of the word

#### Extra Metacharacters to join other metacharacters

[]=set of metacharacters
.= one character
| = or
()= used to group quantifiers

### Now we talk about literals

The normal literals are 'helo' and 'bye', but what if
I want the literal matching of . or * then I need to use \.
"""

string="The sidebar includes a Cheatsheet, full Reference, and Help. You can also save & Share with the Community, and view patterns you create or favorite in My Patterns."

pattern='[A-Z]\w+'

re.findall(pattern,string)

string2="Yesterday at the office party, I met the manager of the east coast branch, her phone number is 202–555–0180. I also exchange my number 202–555–0195 with a recruiter."

pattern3=r'[0-9]{3}–[0-9]{3}–[0-9]{4}'

re.findall(pattern3,string2)

string3="Python 3.0 was released on 03–12–2008. It was a significant revision of the language that is not completely backward-compatible. Many of its major features were backported to Python 2.6. and 2.7 versions that were released on 03.10.2008 and 03/07/2010, respectively."

pattern='[0-9]{2}-[0-9]{2}

#Extract Url

# Use raw strings (r'') to avoid having to escape the forward slashes in the URL pattern
url_pattern = r'https?://[\w.]+'

text = "I love https://www.w3resource.com/python-exercises/re/#EDITOR, but also check out http://facebook.com"

# Use re.findall() to extract all matches of the pattern in the text
matches = re.findall(url_pattern, text)

#extract Named entities with initial capitalized
text = "Sachin Tendulkar was a masterclass, specially his century at Eden Gardens"

individual_word = re.compile(r'[A-Z][a-z]+')
combined_word = re.compile(r'\bUnited States\b')

individual_word.findall(text)

!pip install python-docx

from docx import Document

doc = Document('/content/Manuj Arora_rs.docx')
text = []
for para in doc.paragraphs:
    text.append(para.text)
resume_text = '\n'.join(text)

# Define the regular expressions to match different types of information
email_regex = re.compile(r'\s*([\w\.-]+)@([\w\.-]+)')
date_regex = re.compile(r'([1-9])[- /.](0[1-9]|[12][0-9]|3[01])[- /.](19|20)\d\d')


email_match = email_regex.search(resume_text)
date_match = date_regex.search(resume_text)
