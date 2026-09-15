"""Empirical 8-class Intent Taxonomy definitions and pattern matchers."""

import re
from typing import Dict, List, Tuple, Any
from src.models import IntentCode

# Keyword and regex patterns for each empirical intent category.
# Primary patterns represent core symptoms, problem descriptions, and explicit actions (Weight 2.5).
# Secondary patterns represent contextual entities, background state, or incidental mentions (Weight 0.8).
INTENT_DEFINITIONS: Dict[IntentCode, Dict[str, Any]] = {
    IntentCode.INT_IOS: {
        "name": "iOS & System Updates",
        "description": "Operating system update failures, bricked devices, iOS restore errors, rollback requests, keyboard and autocorrect glitches",
        "primary_keywords": [
            r"update fail(ed|ing)?", r"unable to (update|install)", r"can'?t (update|install)",
            r"stuck on (apple logo|update|loading|black screen|spinning wheel)", r"\bbricked?\b", r"apple logo",
            r"\brestore error\b", r"itunes error", r"\bdowngrade\b", r"revert to (old |previous )?ios",
            r"go back to (old |previous )?ios", r"software update", r"install error",
            r"\bios bug\b", r"\bos update\b",
            # Autocorrect, typing, and keyboard glitch patterns
            r"\bautocorrect\b", r"\bauto-correct\b", r"\bauto correct\b",
            r"predictive text", r"text replacement",
            r"keyboard (glitch|bug|lag|issue|problem|freeze|acting up|broken|messed)",
            r"typing (glitch|bug|lag|issue|problem|acting up|broken|messed)",
            r"letter glitch", r"symbol glitch", r"question marks? (glitch|bug|issue|box|symbol)",
            r"letter [iI]\b", r"the letter i",
            r"exclamation (point|mark)",
            r"symbols? (are )?(glitch|glitching|changing|messed)",
            r"typing (\w+ )?(saying|turning into|changing into|producing) [iI]\.?[tT]",
            r"\b[iI][\s\ufe0f]*[?]\b", r"[\u201c\u201d\"']I[\ufe0f\s]*[\u201c\u201d\"']",
            r"phone (glitch(ing|ed|es)?|freez(ing|es)?|crashing) (after|since) (the )?update",
            r"update (broke|ruined|messed up|screwed up|destroyed)",
            r"(this|the|new|latest) update (broke|ruined|messed|destroyed)",
            r"update to (ios )?11(\.[0-9]+)*",
            r"ios 11(\.[0-9]+)*",
            r"black out with a (white )?loading",
            r"apps? (are )?(crashing|freezing) (after|since) update",
            r"updated? my (phone|iphone|ipad|device)"
        ],
        "secondary_keywords": [
            r"\bios\b", r"\bupdate\b", r"\bupdating\b", r"\binstall\b", r"\binstalling\b",
            r"\brestore\b", r"\bkeyboard\b", r"\btyping\b", r"\bglitch(es|ed|ing)?\b",
            r"\bfirmware\b", r"\bbug(s|gy)?\b"
        ],
        "keywords": [
            r"update fail", r"unable to update", r"bricked", r"apple logo", r"restore error",
            r"itunes error", r"software update", r"ios", r"update", r"restore", r"autocorrect", r"keyboard glitch"
        ]
    },
    IntentCode.INT_STORE: {
        "name": "Account, Store & Billing",
        "description": "App Store downloads, Apple ID sign-in, billing, refunds, subscriptions, disabled accounts, Apple Pay, digital media purchases",
        "primary_keywords": [
            r"app store", r"apple id", r"\bpassword\b", r"\bpurchase\b", r"\breceipt\b",
            r"\brefund(ed)?\b", r"\bbilling\b", r"\bsubscription\b",
            r"charged (again|twice|extra|for|\$\d+|unauthorized|me)", r"unauthorized charge",
            r"overcharged", r"charge on my (card|account)", r"download(ing)? (an )?app",
            r"in-app purchase", r"itunes store", r"card declined", r"payment method",
            # Apple Pay, Wallet, and media purchases
            r"apple pay\b", r"apple music\b", r"itunes (app|music|purchase|account|library|gift card)",
            r"bought (an? )?(app|music|song|album|movie|subscription)",
            r"pay for (this|my|the) (music|app|subscription|service)",
            r"double click (the )?(side|button) (to pay|for apple pay)?",
            r"wallet app\b", r"can'?t buy (music|songs?|apps?)",
            r"purchased? (music|songs?|items?|content|show) (missing|gone|not showing|cannot access)",
            r"recycle an? apple device", r"trade in"
        ],
        "secondary_keywords": [
            r"\baccount\b", r"\bpaid\b", r"\bmusic\b", r"\bwallet\b", r"\bbuy\b", r"\bbought\b"
        ],
        "keywords": [
            r"app store", r"apple id", r"password", r"purchase", r"refund", r"billing", r"subscription", r"download", r"apple pay", r"itunes"
        ]
    },
    IntentCode.INT_ICLOUD: {
        "name": "iCloud & Backup Sync",
        "description": "iCloud drive, photo sync, backup failures, storage quota full",
        "primary_keywords": [
            r"\bicloud\b", r"\bphotos?\b", r"\bbackup\b", r"\bsync(ing|ed)?\b",
            r"storage full", r"iCloud (storage|space|drive)", r"photo library",
            r"synced contacts", r"camera roll", r"icloud photo",
            r"photos? (are )?(not loading|disappeared|missing|deleted) (from|in)? icloud",
            r"recover (my )?photos?"
        ],
        "secondary_keywords": [
            r"\bstorage\b", r"\bdrive\b", r"\bspace\b", r"\bcloud\b"
        ],
        "keywords": [
            r"icloud", r"photos", r"backup", r"sync", r"storage full", r"iCloud space"
        ]
    },
    IntentCode.INT_HARDWARE: {
        "name": "Hardware, Audio & Display",
        "description": "Physical screen, touch unresponsiveness, speaker, microphone, camera, headphone jack, physical buttons",
        "primary_keywords": [
            r"\bscreen\b", r"\bspeaker\b", r"\bmic\b", r"\bmicrophone\b", r"\bcamera\b",
            r"\bcracked\b", r"\btouch(screen)?\b", r"\bdisplay\b", r"\bheadphone\b",
            r"\baudio\b", r"\bvolume button\b", r"\bpower button\b", r"3d touch",
            r"broken screen", r"screen unresponsiv",
            # Audio, alarm, pocket dial, speaker glitches
            r"pocket dial(ing|l)?\b", r"pocket-dial(ing|l)?\b",
            r"alarm sound( not working)?\b", r"silent switch\b", r"speaker(s)? crackl(ing|e)?\b",
            r"crackling (speaker|audio|sound)\b",
            r"earpiece\b", r"home button\b", r"face id\b", r"touch id\b", r"proximity sensor\b",
            r"black screen and (won'?t|will not) turn on",
            r"screen (went|goes) blank"
        ],
        "secondary_keywords": [
            r"\bvolume\b", r"\bbutton\b", r"\bsound\b", r"\bhaptic\b", r"\balarm\b", r"\bsilent\b"
        ],
        "keywords": [
            r"screen", r"speaker", r"mic", r"microphone", r"camera", r"cracked", r"touch", r"display", r"headphone", r"audio"
        ]
    },
    IntentCode.INT_CONN: {
        "name": "Connectivity & Bluetooth",
        "description": "Wi-Fi disconnections, cellular 'No Service', Bluetooth accessory pairing, AirDrop, dropped calls, voicemail",
        "primary_keywords": [
            r"\bwifi\b", r"wi-fi", r"\bbluetooth\b", r"\bairdrop\b", r"no service",
            r"\bcellular\b", r"\bsim card\b", r"\bsim\b", r"cell signal",
            r"connect to (wifi|bluetooth)", r"pairing",
            r"call(s)? dropping\b", r"dropped calls?\b", r"call receiving (issue|problem)\b",
            r"can'?t (make|receive) calls?\b", r"imessage(s)? (not sending|not working|failing|freezing)\b",
            r"can'?t send (or receive )?imessage", r"airdrop (not working|failing|failed)\b",
            r"voicemail (screen|not working|error)\b",
            r"wifi (dropping|disconnecting|greyed out)\b", r"searching for service\b"
        ],
        "secondary_keywords": [
            r"\bsignal\b", r"\bconnect(ion|ing|ed)?\b", r"\bnetwork\b", r"\bcarrier\b",
            r"\bcalls\b", r"\bimessage\b", r"\btexting\b", r"\bvoicemail\b"
        ],
        "keywords": [
            r"wifi", r"wi-fi", r"bluetooth", r"airdrop", r"no service", r"cellular", r"sim", r"signal", r"connect"
        ]
    },
    IntentCode.INT_BATTERY: {
        "name": "Battery & Performance",
        "description": "Rapid battery discharge, overheating, unexpected shutdowns, laggy performance",
        "primary_keywords": [
            r"\bbattery\b", r"\bdrain(ing|s)?\b", r"battery life", r"battery drain",
            r"battery consumption", r"\bcharg(e|ing|er)\b", r"\boverheat(ing)?\b",
            r"battery percentage", r"shut(ting)? down", r"phone died", r"randomly die",
            r"battery soc", r"wastes? (my )?battery",
            r"dies within (an? )?(hour|minutes?|[0-9]+ hours?)\b",
            r"battery dies at [0-9]+%?\b",
            r"dies at [0-9]+%?\b",
            r"battery drain(s|ing)? (fast|rapidly|quick|quickly|crazy)\b",
            r"slow charging\b", r"charging (slowly|slow)\b"
        ],
        "secondary_keywords": [
            r"\bpower\b", r"\blag(gy)?\b", r"\bslow(er)?\b", r"freez(ing|e)\b",
            r"performance\b"
        ],
        "keywords": [
            r"battery", r"drain", r"charging", r"overheating", r"percentage", r"shut down", r"lag", r"slow"
        ]
    },
    IntentCode.INT_WATCH_MAC: {
        "name": "Mac & Watch Ecosystem",
        "description": "Apple Watch pairing/syncing, MacBook hardware/software, macOS issues, Watch activity",
        "primary_keywords": [
            r"apple watch", r"watchos", r"iwatch", r"\bmacbook( pro| air)?\b", r"\bimac\b",
            r"\bmacos\b", r"series [1-9]", r"watch (series|app|sync|face|band|os)",
            r"apple watch (activity|workout|badge|sync)", r"macbook (startup|audio|sound|battery)",
            r"\bmac mini\b", r"\bmac pro\b"
        ],
        "secondary_keywords": [
            r"\bwatch\b", r"\bmac\b"
        ],
        "keywords": [
            r"apple watch", r"watchos", r"macbook", r"imac", r"macos", r"series 1"
        ]
    },
    IntentCode.INT_OUT_OF_SCOPE: {
        "name": "Out-of-Scope / Unclear",
        "description": "General rants, non-actionable tweets, missing context, non-Apple support topics",
        "primary_keywords": [],
        "secondary_keywords": [],
        "keywords": []
    }
}

