import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import re
import string
from sklearn.preprocessing import LabelEncoder

# Constructed with analysis of embedding coverage
misspell_dict = {"cest":"c'est",
                "desaturation": "désaturation",
                "controlaterale":"controlatérale",
                '\x9cdème':"oedème",
                '\x9cdèmes':"oedèmes",
                'léchographie': "échographie",
                "desaturation":"désaturation",
                '\x9cil':"oeil",
                "limplant":"implant",
                "hemorragiques":"hémorragiques",
                "hypoglycemie":"hypoglycémie",
                "lautomate" : 'automate',
                'l\x9cil' : 'oeil',
                'adenopathie':'adénopathie',
                'prothetique':'prothétique',
                'inapproprie': "inapproprié",
                'lartère':'artère',
                'asthenie':'asthénie',
                'man\x9cuvre': 'manoeuvre',
                'lexplantation': 'explantation',
                'lymphoree':'lymphorée',
                'salpyngectomie':'salpingectomie',
                'burnaout':'burnout',
                'lnterventlon': 'intervention',
                'pericardique': 'péricardique',
                'lendométriose':'endométriose',
                'daudition': 'audition',
                'désaltére': 'désaltéré',
                'cephalee':'céphalée',
                'salpaginctomie': 'salpingectomie',
                'menauposée':'ménopausée',
                'deczéma':'eczéma',
                'peritonite': 'péritonite',
                'lablation':'ablation',
                'microjyste': 'microkyste',
                'généralié': 'généralité',
                'débriété': 'ébriété',
                'acidocetose': 'acidocétose',
                'dhéparine':'héparine',
                'dincident':'incident',
                'daiguille':'aiguille',
                'materiovigilance':'matériovigilance',
                'adenomyose': 'adénomyose'
                    }


def replace_typical_misspell(text: str) -> str:
    misspell_re = re.compile('(%s)' % '|'.join(misspell_dict.keys()))

    def replace(match):
        return misspell_dict[match.group(0)]

    return misspell_re.sub(replace, text)


puncts = ['"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']',
          '>', '%', '=', '#', '*', '+', '\\', '•', '~', '@', '£', '·', '_', '{', '}', '©',
          '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█',
          '½', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶',
          '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼',
          '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', '¯', '♦', '¤', '▲',
          '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪',
          '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√']

bad_char = ['\x85']

carriage = [r"\\t|\\n|\\r", "\t|\n|\r", "\r\r\n"]


def clean_text(text: str) -> str:

    text = str(text)
    text = text.replace('\x9c', 'oe')
    text = text.replace('\x92', "'")
    for carr in carriage:
        if carr in text:
            text = text.replace(carr, ' ')
    for bc in bad_char:
        if bc in text:
            text = text.replace(bc, '')
    for punct in puncts:
         if punct in text:
            text = text.replace(punct, '')

    return text


def clean_numbers(text: str) -> str:
    return re.sub(r'\d+', '', text)


def preprocess_text(text: str) -> str:
    if isinstance(text, str):
        text = text.lower()
        text = clean_text(text)
        text = clean_numbers(text)
        text = replace_typical_misspell(text)
        text = text.strip()
        text = re.sub(' +', ' ', text)

    else:
        text = ""
    return text

# Create train and test


X_cols = ['NUMERO_DECLARATION','LIBELLE_COMMERCIAL','DESCRIPTION_INCIDENT', 'ETAT_PATIENT', 'FABRICANT', 'DISTRIBUTEUR']
y_col = 'TEF_ID'

df = pd.read_csv('data/declaration_mrv.csv', sep=";", encoding='latin1')

# Drop all nan values in the target

df = df.dropna(subset=[y_col])

# Fill na values in text fiels with vide

df['DESCRIPTION_INCIDENT'] = df['DESCRIPTION_INCIDENT'].fillna("")
df['ETAT_PATIENT'] = df['TYPE_EFFET'].fillna("")
df['LIBELLE_COMMERCIAL'] = df['LIBELLE_COMMERCIAL'].fillna("")
df['FABRICANT'] = df['FABRICANT'].fillna("")
df['DISTRIBUTEUR'] = df['DISTRIBUTEUR'].fillna("")

train = df[X_cols+[y_col]]

df_subset = train.groupby("TEF_ID").filter(lambda x: len(x) > 15)
df_subset['produit'] = df_subset['LIBELLE_COMMERCIAL'] #+' '+df['FABRICANT']
df_subset['incident'] = df_subset['DESCRIPTION_INCIDENT']

# Encode label
le = LabelEncoder()

df_subset.TEF_ID = le.fit_transform(df_subset.TEF_ID.values)

print("max", df_subset.TEF_ID.max())
print("min", df_subset.TEF_ID.min())


train_index, test_index = next(GroupShuffleSplit(random_state=1029).split(df_subset, groups=df_subset['DESCRIPTION_INCIDENT']))

train = df_subset.iloc[train_index]
test = df_subset.iloc[test_index]

# Clean
text_columns = ['produit', 'incident']
train.loc[:, text_columns] = train.loc[:, text_columns].applymap(preprocess_text)
test.loc[:, text_columns] = test.loc[:, text_columns].applymap(preprocess_text)

train = train[['produit', 'incident', 'TEF_ID']]
test = test[['produit', 'incident', 'TEF_ID']]

print(train.head())
train.to_csv('data/train_test/train.csv', index=False, sep='|')
test.to_csv('data/train_test/test.csv', index=False, sep='|')
