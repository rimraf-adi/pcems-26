import os
import time
import subprocess

paper_dir = "/Users/adityakinjawadekar/Documents/100xcode/pcems26/paper"
files_to_watch = [
    os.path.join(paper_dir, "content.tex"),
    os.path.join(paper_dir, "springer.tex"),
    os.path.join(paper_dir, "references.bib"),
]

def get_mtimes():
    return [os.path.getmtime(f) for f in files_to_watch if os.path.exists(f)]

def compile_springer():
    print("[Watcher] Compiling springer.pdf with full bibtex resolution...", flush=True)
    subprocess.run(["xelatex", "-interaction=nonstopmode", "springer.tex"], cwd=paper_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["bibtex", "springer"], cwd=paper_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["xelatex", "-interaction=nonstopmode", "springer.tex"], cwd=paper_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["xelatex", "-interaction=nonstopmode", "springer.tex"], cwd=paper_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("[Watcher] Done! springer.pdf updated with fully rendered citations.", flush=True)

last_mtimes = get_mtimes()
compile_springer()

while True:
    time.sleep(1.5)
    current_mtimes = get_mtimes()
    if current_mtimes != last_mtimes:
        last_mtimes = current_mtimes
        compile_springer()
