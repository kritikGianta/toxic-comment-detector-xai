"""
Configuration file for Toxic Comment Detector
All constants, settings, and replacement dictionaries
"""

# Model Configuration
MODEL_PATH = "./exported_model"
MAX_LENGTH = 128
DEFAULT_THRESHOLD = 0.61  # Optimal threshold from training
DEVICE = "cpu"

# XAI Configuration
IG_STEPS = 50
ATTRIBUTION_THRESHOLD = 0.15

# Rewrite Configuration
MAX_REWRITE_PASSES = 2
MIN_IMPROVEMENT = 0.01

# Profanity Replacements
PROFANITY_REPLACEMENTS = {
    "fuck": "",
    "fucking": "",
    "shit": "",
    "bullshit": "nonsense",
    "asshole": "person",
    "bitch": "person",
    "bastard": "person",
    "dumbass": "person",
    "jerk": "person",
    "moron": "person",
    "retard": "person",
    "hell": "",
    "damn": "",
    "crap": "",
    "piss": "",
}

# Insult Replacements
INSULT_REPLACEMENTS = {
    "idiot": "person",
    "stupid": "unwise",
    "dumb": "uninformed",
    "pathetic": "unhelpful",
    "useless": "not effective",
    "terrible": "not good",
    "awful": "unpleasant",
    "horrible": "poor",
    "worst": "not ideal",
    "lazy": "unmotivated",
    "loser": "person",
    "fool": "person",
}

# Hate Speech Replacements
HATE_REPLACEMENTS = {
    "hate": "dislike",
    "despise": "strongly dislike",
    "loathe": "do not like",
    "can't stand": "find difficult",
    "sick of": "tired of",
}

# Softening Phrases (prepend to text)
SOFTENING_PHRASES = [
    "I think",
    "it seems that",
    "in my opinion",
    "from my perspective",
    "I feel that",
    "it might be better to",
]

# Safe Phrase Replacements
SAFE_PHRASE_REPLACEMENTS = [
    "please be respectful",
    "this is inappropriate",
    "I strongly disagree",
    "let's keep this respectful",
    "I don't agree with this",
    "this could be handled better",
    "this feels unfair",
    "I think this is not constructive",
    "",
]

# Command Phrase Replacements
COMMAND_REPLACEMENTS = [
    "please leave",
    "let's keep this respectful",
    "I strongly disagree",
    "this feels inappropriate",
    "this conversation should stay respectful",
]

# Words for toxicity detection
PROFANITY_WORDS = {
    "fuck", "shit", "asshole", "bitch", "bastard",
    "damn", "hell", "bullshit", "jerk", "crap"
}

INSULT_WORDS = {
    "idiot", "stupid", "dumb", "moron",
    "pathetic", "useless", "lazy", "worst", "loser"
}

COMMAND_WORDS = {
    "get", "leave", "go", "shut", "die"
}

# Bias analysis templates
BIAS_TEMPLATES = [
    "I am a {}",
    "This person is a {}",
    "The individual identifies as {}",
    "They are a {}"
]

IDENTITY_GROUPS = {
    "Gender": ["man", "woman", "non-binary person"],
    "Religion": ["Christian", "Muslim", "Hindu", "Jewish", "Buddhist"],
    "Nationality": ["Indian", "American", "British", "Chinese", "Mexican"],
    "Sexual Orientation": ["gay person", "straight person", "bisexual person"],
    "Other": ["student", "engineer", "teacher", "doctor"]
}

# Severity levels
def get_severity(score):
    """Return severity level based on toxicity score"""
    if score < 0.3:
        return "Non-toxic", "green"
    elif score < 0.5:
        return "Mild", "yellow"
    elif score < 0.7:
        return "Moderate", "orange"
    elif score < 0.85:
        return "Severe", "red"
    else:
        return "Extreme", "darkred"

# Example texts for quick testing
EXAMPLE_TEXTS = {
    "Toxic - Profanity": "You are a fucking idiot, shut up!",
    "Toxic - Insult": "This is the dumbest thing I've ever seen, you're pathetic",
    "Toxic - Hate": "I hate you and everything you stand for",
    "Borderline": "This is really stupid and terrible",
    "Non-toxic - Positive": "Thank you so much for your help, this is wonderful!",
    "Non-toxic - Neutral": "I disagree with your opinion on this matter",
}
