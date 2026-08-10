import argparse
import re
import os

def strip_latex(text: str) -> str:
    """
    Strips LaTeX formatting tags to extract plain text.
    """
    # 1. Remove comments
    text = re.sub(r'%.*', '', text)
    
    # 2. Extract text from formatting commands like \textbf{...}, \textit{...}, \section{...}
    # This simple regex handles one level of braces.
    text = re.sub(r'\\(?:section|subsection|subsubsection|textbf|textit|emph|textcolor|underline)\{([^}]*)\}', r'\1', text)
    
    # 3. Completely remove certain commands and their arguments (e.g., \cite{...}, \ref{...}, \label{...})
    text = re.sub(r'\\(?:cite|ref|label|includegraphics|caption)\{[^}]*\}', '', text)
    
    # Strip entire table and tabular environments (must be done before stripping general \begin and \end)
    text = re.sub(r'\\begin\{(table|table\*|tabular|tabularx|longtable|tabu|tabular\*)\}.*?\\end\{\1\}', '', text, flags=re.DOTALL)
    
    # 4. Remove environments like \begin{...} and \end{...}
    text = re.sub(r'\\begin\{[^}]+\}', '', text)
    text = re.sub(r'\\end\{[^}]+\}', '', text)
    
    # 5. Remove any remaining commands (e.g., \maketitle, \newpage, \hline)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    
    # 6. Remove stray braces and table alignment characters (&, \\)
    text = text.replace('{', '').replace('}', '').replace('&', ' ')
    
    # 7. Clean up whitespace (multiple spaces to single space, multiple newlines to double newline)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text.strip()

def chunk_text(text: str, chunk_size: int = 2000) -> str:
    """Chunks text into paragraphs of up to chunk_size characters, breaking at word boundaries."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1
            
    if current_chunk:
        chunks.append(' '.join(current_chunk))
        
    return '\n\n'.join(chunks)

def main():
    parser = argparse.ArgumentParser(description="Convert a LaTeX (.tex) file to a plain text (.txt) file.")
    parser.add_argument("input_file", help="Path to the input .tex file")
    parser.add_argument("-o", "--output_file", help="Path to the output .txt file (Optional, defaults to input filename with .txt)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        return
        
    with open(args.input_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
    plain_text = strip_latex(content)
    chunked_text = chunk_text(plain_text, 2000)
    
    # Determine output file path
    out_path = args.output_file
    if not out_path:
        base_name = os.path.splitext(args.input_file)[0]
        out_path = base_name + ".txt"
        
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(chunked_text)
        
    print(f"✅ Successfully converted '{args.input_file}' to plain text.")
    print(f"📄 Output saved to: {out_path}")

if __name__ == "__main__":
    main()
