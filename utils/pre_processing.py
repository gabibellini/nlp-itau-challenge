import re
import unicodedata
import pandas as pd
from functools import reduce 

#--- 1. Pre-processing Functions ---

def fix_special_characters(text):
  """ Remove accents and uniform unicode characters """
  if not isinstance(text, str):
    return ''
  return unicodedata.normalize('NFKD', text).encode('ascii','ignore').decode('utf-8')

def lowering(text):
  """ Convert text to lower case """
  if not isinstance(text,str):
    return ''
  return text.lower()

def remove(text,regex):
  """ Remove unwanted patterns with regex """
  if not isinstance(text,str):
    return ''
  return regex.sub(' ', text)

def substitute_spaces(text,regex):
  """Substitute multiple spaces for unique space """
  if not isinstance(text,str):
    return ''
  return regex.sub(' ', text)

#--- 2. Functions Pipeline ---

def apply_pipeline(text,funcs):
  """ Apply sequence of functions to a text """
  return reduce(lambda t,f: f(t), funcs, text)


#--- 3. Main Pre-processing Function (multiple columns) ---

def preprocess_columns(df, columns, funcs):
  """ Apply functions pipeline to multiple columns of a DataFrame.

  Args:
    df (pd.DataFrame): dataset
    columns (list): list of columns to be processed
    funcs (list): list of functions to apply in sequence

  Return:
    pd.Dataframe: dataframe with modified columns """

  for col in columns:
    df[col] = df[col].apply(lambda x: apply_pipeline(x, funcs)).str.strip()
  return df


# --- 4. Others ---


# Regex
regex_remove = re.compile(r'[^a-z0-9\s&/-]')  # mantém &, / e -
regex_spaces = re.compile(r'\s+')
regex_enterprise = re.compile(r'\b(ltda|eireli|s/a|me|ei|comercial|comercio)\b')



# --- 5. Columns to be processed ---

y_columns = ['nome_fantasia', 'razaosocial']
uf_column = ['uf']


# --- 6. Pipeline sequence

funcs = [
    fix_special_characters,
    lowering,
    lambda x: remove(x, regex_remove),
    lambda x: remove(x, regex_enterprise),
    lambda x: substitute_spaces(x, regex_spaces)
]