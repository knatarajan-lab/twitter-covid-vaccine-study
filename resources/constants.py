from pathlib import Path

from jinja2 import Environment

JINJA_ENV = Environment(
    # block tags on their own lines
    # will not cause extra white space
    trim_blocks=True,
    lstrip_blocks=True,
    # syntax highlighting should be better
    # with these comment delimiters
    comment_start_string='--',
    comment_end_string=' --',
    # in jinja2 autoescape is for html; jinjasql supports autoescape for sql
    # TODO Look into jinjasql for sql templating
    autoescape=False)

# Requests params
STATUS_FORCELIST = [500, 502, 503, 504]
MAX_RETRIES = 5
BACKOFF_FACTOR = 0.5
MAX_TIMEOUT = 62

PFIZER = 'pfizer'
BIONTECH = 'biontech'

MODERNA = 'moderna'

MRNA = 'mrna'

JANSSEN = 'janssen'
JNJ = 'jnj'
J_AND_J = '"j and j"'
J_N_J = '"j n j"'
JOHNSON_JOHNSON = '"johnson johnson"'
JOHNSON_AND_JOHNSON = '"johnson and johnson"'
JOHNSON_N_JOHNSON = '"johnson n johnson"'
# Twitter tokenizes tweets and cannot recognize punctuation with exact word match
# "J J" would catch "J & J" but also "J_J"
J_J = '"j j"'

ASTRA = 'astra'
ASTRAZENECA = 'astrazeneca'
OXFORD = 'oxford'
ASTRAZENECA_OXFORD = '"astrazeneca oxford"'

PFIZER_KEYWORDS = [PFIZER, BIONTECH]
MODERNA_KEYWORDS = [MODERNA]
JNJ_KEYWORDS = [
    JANSSEN, JNJ, J_AND_J, J_N_J, JOHNSON_JOHNSON, JOHNSON_AND_JOHNSON,
    JOHNSON_N_JOHNSON, J_J
]
ASTRAZENECA_KEYWORDS = [ASTRA, ASTRAZENECA, OXFORD, ASTRAZENECA_OXFORD]

US_VACCINE_KEYWORDS = [
    PFIZER_KEYWORDS,
    MODERNA_KEYWORDS,
    JNJ_KEYWORDS
]
VACCINE_KEYWORDS = [
    PFIZER_KEYWORDS, MODERNA_KEYWORDS, JNJ_KEYWORDS, ASTRAZENECA_KEYWORDS
]
MAX_TWEETS = 100
TWEET_FIELDS = [
    'author_id', 'created_at', 'public_metrics', 'id', 'text', 'geo', 'in_reply_to_user_id', 'conversation_id'
]
USER_FIELDS = [
    'id', 'username', 'name', 'location', 'public_metrics', 'verified'
]
PLACE_FIELDS = [
    'id', 'full_name', 'contained_within', 'country', 'country_code', 'name',
    'place_type'
]
EXPANSIONS = [
    'author_id', 'referenced_tweets.id', 'in_reply_to_user_id', 'geo.place_id'
]
HOUR = 60

# Paths
curr_path = Path(__file__)
resources_path = curr_path.parent
base_path = resources_path.parent
core_path = base_path / 'core'
data_path = base_path / 'data'
DATA_COLUMNS = ['id', 'label', 'text']

LABEL_TARGET_MAP = {
    'Y': 0,
    'N': 1,
    'S': 2,
    'A': 3,
}

PUNCTS = r'#"\\\$%\(\)\*\+,/:;<=>@\[\]\^_\{|\}~`'
