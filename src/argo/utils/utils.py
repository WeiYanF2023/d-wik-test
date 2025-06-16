from __future__ import annotations

import re # Added for tag parsing
from typing import List, Optional

# --- Utility Functions ---

def extract_tag_content(text: str, tag: str) -> Optional[str]:
    """Extracts content from all occurrences of <tag>content</tag>."""
    # Use re.IGNORECASE for case-insensitive matching
    all_matches = re.findall(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
    if all_matches:
        # Return as a single string, joining all occurrences, stripping whitespace
        return "\n".join(match.strip() for match in all_matches).strip()
    else:
        return None

def check_tag_presence(text: str, tag: str) -> bool:
    """Checks if <tag/> or <tag>...</tag> exists."""
    # Use re.IGNORECASE for case-insensitive matching
    # Match self-closing tag <tag/> or opening tag <tag>
    return bool(re.search(rf"<{tag}\s*/>|<{tag}>", text, re.IGNORECASE))