
import os
import urllib.request

LOCAL_ZIP = "reinforcement.zip"
ORIGINAL_URLOPEN = urllib.request.urlopen


def fake_urlopen(url, *args, **kwargs):
    url_str = url.full_url if hasattr(url, "full_url") else str(url)
    print("urlopen URL:", url_str)

    if "reinforcement.zip" in url_str:
        if not os.path.isfile(LOCAL_ZIP):
            raise FileNotFoundError(
                f"Expected local file '{LOCAL_ZIP}' in the current directory."
            )
        print(">>> Intercepting download. Returning local reinforcement.zip instead...")
        return open(LOCAL_ZIP, "rb")

    return ORIGINAL_URLOPEN(url, *args, **kwargs)


def main():
    script_name = "submission_autograder.py"
    script_path = os.path.abspath(script_name)

    if not os.path.isfile(script_name):
        raise FileNotFoundError(f"Could not find {script_name} in this directory.")

    if not os.path.isfile(LOCAL_ZIP):
        raise FileNotFoundError(f"Could not find {LOCAL_ZIP} in this directory.")

    urllib.request.urlopen = fake_urlopen

    with open(script_name, "r", encoding="utf-8") as f:
        code = compile(f.read(), script_name, "exec")
        exec_globals = {
            "__name__": "__main__",
            "__file__": script_path,
        }
        exec(code, exec_globals)


if __name__ == "__main__":
    main()