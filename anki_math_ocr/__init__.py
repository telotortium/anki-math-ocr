# For docs, see ../setup.py
import argparse
import importlib
import json
import logging
import os
import os.path
import pdb
import re
import shutil
import subprocess
import sys
import traceback
from itertools import islice

import google.generativeai as genai
import PIL.Image
import requests

# Create logger that logs to standard error
logger = logging.getLogger("anki-math-ocr")
# These 2 lines prevent duplicate log lines.
logger.handlers.clear()
logger.propagate = False

LEVEL_DEFAULT = logging.INFO
level = os.environ.get("ANKI_MATH_OCR_LOGLEVEL")
if level:
    level = level.upper()
else:
    level = LEVEL_DEFAULT
logger.setLevel(level)

# Create handler that logs to standard error
handler = logging.StreamHandler()
handler.setLevel(level)

# Create formatter and add it to the handler
formatter = logging.Formatter("[%(levelname)8s %(asctime)s - %(name)s] %(message)s")
handler.setFormatter(formatter)

# Add handler to the logger
logger.addHandler(handler)

ANKICONNECT_URL_DEFAULT = "http://localhost:8765"
ankiconnect_url = os.environ.get(
    "ANKI_MATH_OCR_ANKICONNECT_URL", ANKICONNECT_URL_DEFAULT
)
ANKICONNECT_VERSION = 6

# Load secrets from pockexport for use by pocket module.
loader = importlib.machinery.SourceFileLoader(
    "secrets", os.path.expanduser("~/.config/anki-math-ocr/secrets.py")
)
spec = importlib.util.spec_from_loader("secrets", loader)
secrets = importlib.util.module_from_spec(spec)
loader.exec_module(secrets)


def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def ankiconnect_request(payload):
    payload["version"] = ANKICONNECT_VERSION
    logger.debug("payload = %s", payload)
    response = json.loads(requests.post(ankiconnect_url, json=payload, timeout=3).text)
    logger.debug("response = %s", response)
    if response["error"] is not None:
        logger.warning("payload %s had response error: %s", payload, response)
    return response


def anki_sync():
    logger.info("Syncing Anki")
    return ankiconnect_request({"action": "sync"})


BATCH_SIZE = 5


_ANKI_MEDIA_PATH = None


def media_path():
    global _ANKI_MEDIA_PATH
    if _ANKI_MEDIA_PATH is None:
        _ANKI_MEDIA_PATH = ankiconnect_request({"action": "getMediaDirPath"})["result"]
    return _ANKI_MEDIA_PATH


IMG_SRC_RE = re.compile(r'< *img +src="(?P<filename>.*?)" *>', re.IGNORECASE)

OCR_COMMENT_IMAGE_SRC_TEMPLATE = """\
<div class="anki-math-ocr">
<p>anki-math-ocr OCR for image:</p>
<p>{ocr_html}</p>
<p><img class="anki-math-ocr-image" src="{filename}"></p>
</div>"""


def format_string_to_regex(format_string):
    """
    Converts a format string with {placeholder} patterns to a regex pattern.

    Args:
    format_string (str): The format string containing {placeholder} patterns.

    Returns:
    str: A regex pattern with capturing groups for the placeholders.
    """
    # Escape the entire string
    regex_pattern = re.escape(format_string)

    # Replace escaped placeholders with a regex group to capture any characters
    # The placeholder is assumed to not contain '}' character inside
    regex_pattern = re.sub(r"\\{([^\\}]+)\\}", r"(?P<\1>.*?)", regex_pattern)
    return re.compile(regex_pattern, re.DOTALL)


# OCR_COMMENT_IMAGE_SRC_TEMPLATE_REGEX = format_string_to_regex(
#     OCR_COMMENT_IMAGE_SRC_TEMPLATE
# )
OCR_COMMENT_IMAGE_SRC_TEMPLATE_REGEX = re.compile(
    r"""<div class="anki-math-ocr">.*?<img\s+class="anki-math-ocr-image"\s+src="(?P<filename>.*?)".*?</div>""",
    re.DOTALL,
)


def ocr_comment_image_src(img_html):
    filename = img_html.group("filename")
    ocr_html = ocr_image(os.path.join(media_path(), filename))
    return OCR_COMMENT_IMAGE_SRC_TEMPLATE.format(ocr_html=ocr_html, filename=filename)


def ocr_comment_image_src_unocr(match):
    filename = match.group("filename")
    return f'<img src="{filename}">'


_OCR_MODEL = None


def ocr_model():
    global _OCR_MODEL
    if _OCR_MODEL is None:
        genai.configure(api_key=secrets.google_api_key)
        _OCR_MODEL = genai.GenerativeModel("gemini-pro-vision")
    return _OCR_MODEL


def ocr_image(file_path):
    """Submits the image at file_path to Google's OCR API and returns HTML for the text.

    Math in the image is converted to LaTeX and put in `\\( ... \\)` and `\\[ ... \\]` tags."""
    img = PIL.Image.open(file_path)
    response = ocr_model().generate_content(
        [
            "Transcribe the following picture to text. Convert any math in the picture to LaTeX",
            img,
        ],
        stream=True,
    )
    response.resolve()

    # Convert response.text, which is in Markdown, to HTML, and add tags for math.
    text = response.text
    logger.debug("response.text = %s", text)
    parts = text.split("$$")
    newparts = []
    for i, part in enumerate(parts):
        if i != 0:
            # It appears that markdown_to_html eliminates one backslash from these patterns, so I need to double the backslashes here.
            newparts.append("\\\\[ " if i % 2 != 0 else " \\\\]")
        newparts.append(part)
    text = "".join(newparts)
    parts = text.split("$")
    newparts = []
    for i, part in enumerate(parts):
        if i != 0:
            # It appears that markdown_to_html eliminates one backslash from these patterns, so I need to double the backslashes here.
            newparts.append("\\\\( " if i % 2 != 0 else " \\\\)")
        newparts.append(part)
    text = "".join(newparts)
    html = markdown_to_html(text.strip())
    return html


def markdown_to_html(md):
    proc = subprocess.Popen(
        [shutil.which("pandoc"), "-fmarkdown", "-thtml"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    (html, _) = proc.communicate(input=md.encode("utf-8"), timeout=5)
    if proc.returncode != 0:
        raise Exception(f"pandoc failed - returncode = {proc.returncode}")
    return html.decode(encoding="utf-8", errors="strict")


def main():
    try:
        _main()
    except Exception:
        debug = os.environ.get("ANKI_MATH_OCR_DEBUG", None)
        if debug and debug != "0":
            _extype, _value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
        else:
            raise


def _main():
    parser = argparse.ArgumentParser(
        prog="anki-math-ocr",
        description="OCR Anki images using Google Gemini Pro",
        epilog=f"""Environment variables:

- ANKI_MATH_OCR_ANKICONNECT_URL: set to the URL of AnkiConnect. Default:
  {ANKICONNECT_URL_DEFAULT}
  set to "{ANKICONNECT_URL_DEFAULT}".
- ANKI_MATH_OCR_DEBUG: set in order to debug using PDB upon .
- ANKI_MATH_OCR_LOGLEVEL: set log level. Default: {LEVEL_DEFAULT}
""",
    )
    parser.add_argument(
        "--query", help="The query to find notes to scan for images.", required=True
    )
    parser.add_argument(
        "--fields", help="Names of fields to scan for images.", required=True
    )
    parser.add_argument(
        "--dry-run", help="Dry-run mode.", action="store_true", required=False
    )
    parser.add_argument(
        "--unocr",
        help="Remove OCR previously added by this script.",
        action="store_true",
        required=False,
    )
    args = parser.parse_args()

    # First, find notes added to Anki but not yet to Pocket and add them to
    # Pocket.
    anki_sync()
    response = ankiconnect_request(
        {
            "action": "findNotes",
            "params": {
                "query": f"{args.query}",
            },
        }
    )
    note_ids = response["result"]
    response = ankiconnect_request(
        {
            "action": "notesInfo",
            "params": {
                "notes": note_ids,
            },
        }
    )
    note_infos = response["result"]
    fields = args.fields.split(",")
    logger.debug(f"fields = {fields}")
    if note_infos:
        try:
            for batch in batched(note_infos, BATCH_SIZE):
                actions = []
                for note_info in batch:
                    new_fields = {}
                    for field in fields:
                        if field not in note_info["fields"]:
                            continue
                        content = note_info["fields"][field]
                        if not content:
                            continue
                        if args.unocr:
                            result = OCR_COMMENT_IMAGE_SRC_TEMPLATE_REGEX.sub(
                                ocr_comment_image_src_unocr, content["value"]
                            )
                            if result != content["value"]:
                                new_fields[field] = result
                        else:
                            try:
                                result = IMG_SRC_RE.sub(
                                    ocr_comment_image_src, content["value"]
                                )
                                if result != content["value"]:
                                    new_fields[field] = result
                            except ValueError:
                                logger.error(
                                    "Could not parse image in field %s of note %s",
                                    field,
                                    note_info["noteId"],
                                )
                                continue
                            except Exception as e:
                                logger.error(
                                    "Exception while parsing image in field %s of note %s: %s",
                                    field,
                                    note_info["noteId"],
                                    e,
                                )
                                continue
                    if not new_fields:
                        continue
                    update_action = {
                        "action": "updateNoteFields",
                        "params": {
                            "note": {
                                "id": note_info["noteId"],
                                "fields": new_fields,
                            },
                        },
                    }
                    if args.dry_run:
                        logger.info(
                            "Would update note %s with fields %s",
                            note_info["noteId"],
                            new_fields,
                        )
                    else:
                        logger.debug(
                            "Updating note %s with fields %s",
                            note_info["noteId"],
                            new_fields,
                        )
                        actions.append(update_action)

                response = ankiconnect_request(
                    {
                        "action": "multi",
                        "params": {"actions": actions},
                    }
                )
        finally:
            anki_sync()


if __name__ == "__main__":
    main()
