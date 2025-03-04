import os
from pygments import highlight
from pygments.lexers import get_lexer_for_filename, PythonLexer
from pygments.formatters import TerminalFormatter
import logging

logger = logging.getLogger(__name__)

def get_supported_extensions():
    """返回支持的文件扩展名列表"""
    return [".py", ".js", ".java", ".cpp", ".c", ".h", ".html", ".css", ".go", ".rs"]

def count_code_files(directory, file_exts=None):
    """计算目录中的代码文件数量"""
    if file_exts is None:
        file_exts = get_supported_extensions()

    count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in file_exts):
                count += 1

    return count

def display_code_snippet(file_path, line_start=None, line_end=None):
    """高亮显示代码片段"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()

        if line_start and line_end:
            lines = code.split('\n')
            code = '\n'.join(lines[line_start-1:line_end])

        try:
            lexer = get_lexer_for_filename(file_path)
        except:
            lexer = PythonLexer()

        formatted_code = highlight(code, lexer, TerminalFormatter())
        return formatted_code
    except Exception as e:
        logger.error(f"无法显示代码: {str(e)}")
        return code
