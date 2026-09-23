import re

# Read both files
with open('paper/main.tex', 'r', encoding='utf-8') as f:
    main = f.read()
with open('paper_arxiv/main.tex', 'r', encoding='utf-8') as f:
    arxiv = f.read()

# Extract body from main (from \begin{abstract} to \end{document})
main_body_match = re.search(r'\\begin\{abstract\}.*?\\end\{document\}', main, re.DOTALL)
if main_body_match:
    main_body = main_body_match.group(0)
else:
    raise ValueError("Could not find body in main.tex")

# Extract arxiv preamble (everything before \begin{abstract})
arxiv_preamble_match = re.search(r'^.*?\\begin\{abstract\}', arxiv, re.DOTALL)
if arxiv_preamble_match:
    arxiv_preamble = arxiv_preamble_match.group(0)
    # Remove trailing \begin{abstract} from preamble to avoid duplication
    arxiv_preamble = arxiv_preamble.rsplit('\\begin{abstract}', 1)[0]
else:
    raise ValueError("Could not find preamble in arxiv")

new_arxiv = arxiv_preamble + main_body

with open('paper_arxiv/main.tex', 'w', encoding='utf-8') as f:
    f.write(new_arxiv)

print("Synced paper_arxiv/main.tex with humanized body")