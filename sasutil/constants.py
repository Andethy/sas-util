REQUIRED = ('no voice', 'no voices', 'no vocal', 'no vocals', 'instrumental')
TAGS = ('X',
        'no voice',
        'singer',
        'duet',
        'plucking',
        'hard rock',
        'world',
        'bongos',
        'harpsichord',
        'female singing',
        'classical',
        'sitar',
        'chorus',
        'female opera',
        'male vocal',
        'vocals',
        'clarinet',
        'heavy',
        'silence',
        'beats',
        'men',
        'woodwind',
        'funky',
        'no strings',
        'chimes',
        'foreign',
        'no piano',
        'horns',
        'classical',
        'female',
        'no voices',
        'soft rock',
        'eerie',
        'spacey',
        'jazz',
        'guitar',
        'quiet',
        'no beat',
        'banjo',
        'electric',
        'solo',
        'violins',
        'folk',
        'female voice',
        'wind',
        'happy',
        'ambient',
        'new age',
        'synth',
        'funk',
        'no singing',
        'middle eastern',
        'trumpet',
        'percussion',
        'drum',
        'airy',
        'voice',
        'repetitive',
        'birds',
        'space',
        'strings',
        'bass',
        'harpsicord',
        'medieval',
        'male voice',
        'girl',
        'keyboard',
        'acoustic',
        'loud',
        'classic',
        'string',
        'drums',
        'electronic',
        'not classical',
        'chanting',
        'no violin',
        'not rock',
        'no guitar',
        'organ',
        'no vocal',
        'talking',
        'choral',
        'weird',
        'opera',
        'soprano',
        'fast',
        'acoustic guitar',
        'electric guitar',
        'male singer',
        'man singing',
        'classical guitar',
        'country',
        'violin',
        'electro',
        'reggae',
        'tribal',
        'dark',
        'male opera',
        'no vocals',
        'irish',
        'electronica',
        'horn',
        'operatic',
        'arabic',
        'lol',
        'low',
        'instrumental',
        'trance',
        'chant',
        'strange',
        'drone',
        'synthesizer',
        'heavy metal',
        'modern',
        'disco',
        'bells',
        'man',
        'deep',
        'fast beat',
        'industrial',
        'hard',
        'harp',
        'no flute',
        'jungle',
        'pop',
        'lute',
        'female vocal',
        'oboe',
        'mellow',
        'orchestral',
        'viola',
        'light',
        'echo',
        'piano',
        'celtic',
        'male vocals',
        'orchestra',
        'eastern',
        'old',
        'flutes',
        'punk',
        'spanish',
        'sad',
        'sax',
        'slow',
        'male',
        'blues',
        'vocal',
        'indian',
        'no singer',
        'scary',
        'india',
        'woman',
        'woman singing',
        'rock',
        'dance',
        'piano solo',
        'guitars',
        'no drums',
        'jazzy',
        'singing',
        'cello',
        'calm',
        'female vocals',
        'voices',
        'different',
        'techno',
        'clapping',
        'house',
        'monks',
        'flute',
        'not opera',
        'not english',
        'oriental',
        'beat',
        'upbeat',
        'soft',
        'noise',
        'choir',
        'female singer',
        'rap',
        'metal',
        'hip hop',
        'quick',
        'water',
        'baroque',
        'women',
        'fiddle',
        'english',
        'X')

TAG_COUNT = len(TAGS) - 2

TAG_MAP = {'rock': 'rock',
           'hard rock': 'rock',
           'metal': 'rock',
           'heavy metal': 'rock',
           'orchestra': 'orchestral',
           'classical': 'orchestral',
           'techno': 'electronic',
           'ambient': 'electronic',
           'house': 'electronic',
           'hip hop': 'hip hop',
           'rap': 'hip hop',
           'calm': 'test/safe',
           'happy': 'test/safe',
           'mellow': 'test/safe',
           'scary': 'test/dangerous',
           'eerie': 'test/dangerous',
           'fast beat': 'test/dangerous',
           'jazz': 'jazz'}

JSON_FIELDS = ('id', 'tags', 'file')

JSON_MFCC_PATH = '../resources/fma/mfcc.json'
JSON_ONSET_PATH = '../resources/fma/onset.json'
JSON_BUCKETS_PATH = '../resources/fma/buckets.json'
ENERGY_FIELD = 'Energy'
MFCC_FIELD = 'MFCCS_BUCKET'
MFCC_FIELDS = ('Name', 'Index', 'Min', 'Max', 'Energy', 'BUCKET')
ONSET_FIELDS = ('Name', 'Index', 'Onsets', 'BUCKET')
OUTPUT_BUCKETS = ("00",
                  "01",
                  "02",
                  "03",
                  "10",
                  "11",
                  "12",
                  "13",
                  "20",
                  "21",
                  "22",
                  "23",
                  "30",
                  "31",
                  "32",
                  "33")
