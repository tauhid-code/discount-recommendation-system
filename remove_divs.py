#!/usr/bin/env python
import re

with open('app/streamlit_app.py', 'r') as f:
    content = f.read()

# Remove the div.card openings and closings
content = re.sub(r"    st\.markdown\('<div class=\"card\">', unsafe_allow_html=True\)\n\s*\n", "", content)
content = re.sub(r"    st\.markdown\('</div>', unsafe_allow_html=True\)\n", "", content)

with open('app/streamlit_app.py', 'w') as f:
    f.write(content)

print('Removed div.card wrappers successfully')
