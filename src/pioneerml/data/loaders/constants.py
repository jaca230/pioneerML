"""
Constants for data loading and classification.
"""

# Bitmasks for particle identification
PION_MASK = 0b00001
MUON_MASK = 0b00010
POSITRON_MASK = 0b00100
ELECTRON_MASK = 0b01000
OTHER_MASK = 0b10000

# Class mapping for group classification
BIT_TO_CLASS = {
    PION_MASK: 0,
    MUON_MASK: 1,
    POSITRON_MASK: 2,  # positron collapses to mip label
    ELECTRON_MASK: 2,  # electron collapses to mip label
    # OTHER_MASK hits are ignored for supervision
}

CLASS_NAMES = {0: 'pion', 1: 'muon', 2: 'mip'}
NUM_GROUP_CLASSES = len(set(BIT_TO_CLASS.values()))

# Node-level classes for splitter
NUM_NODE_CLASSES = 3  # channels: [pion, muon, mip]
NODE_LABEL_TO_NAME = {0: "pion", 1: "muon", 2: "mip"}

# Placeholder: when positron angle labels become available, update extractor
DEFAULT_POSITRON_ANGLE = [0.0, 0.0]

