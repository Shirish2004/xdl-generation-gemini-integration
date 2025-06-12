import os
from google import genai
from google.genai import types
import textwrap  
from verify import verify_xdl  

# Initialize Gemini client once
#genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
#client = genai.GenerativeModel(model_name="gemini-2.0-pro")  # or "gemini-2.0-flash"

def prompt_gemini(input_xdl,description):
    """Calls Gemini with input prompt."""
    full_prompt = textwrap.dedent(f"""
    {description}

    Convert this procedure into valid XDL format:
    {input_xdl}
    """).strip()

    try:
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        response = client.models.generate_content(model="gemini-2.0-flash",contents=[full_prompt])
        return response.text
    except Exception as e:
        raise RuntimeError(f"Gemini model call failed: {e}")

def translate(input_xdl, verbose=False):
    """Translate input description into valid XDL using Gemini + verify."""

    if verbose:
        print("Start translating the following input into XDL:", input_xdl)

    #if model not in ["gemini-pro", "gemini-2.0-pro", "gemini-2.0-flash"]:
     #   raise ValueError("Unsupported model for Gemini. Use gemini-pro or gemini-2.0-*")

    if "GOOGLE_API_KEY" not in os.environ:
        raise RuntimeError("GOOGLE_API_KEY is not set in environment variables.")

    # Load XDL format guide
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "XDL_description.txt")
    with open(filename, "r") as f:
        XDL_description = f.read()

    correct_syntax = False
    errors = {}
    prev_input_xdl = input_xdl

    for step in range(10):
        if verbose:
            print(f"# Step {step}")
            print("Instruction to Gemini ------------------------")
            print(f"Convert to XDL: {input_xdl}")
            print("--------------------------------------------------\n")

        try:
            model_output = prompt_gemini(input_xdl, XDL_description)
        except Exception as e:
            print(f"Error during model call: {e}")
            break

        # Check for <XDL> structure
        if "<XDL>" in model_output:
            model_output = model_output[model_output.index("<XDL>"):model_output.rindex("</XDL>") + 6]

            if verbose:
                print("Output of Gemini --------------------------------")
                print(model_output)
                print("-------------------------------------------------\n")

            compile_errors = verify_xdl(model_output)
            errors[step] = {
                "errors": compile_errors,
                "input_xdl": input_xdl,
                "model_output": model_output,
            }

            if not compile_errors:
                correct_syntax = True
                break
            else:
                # Gather all unique error messages
                error_list = set()
                for item in compile_errors:
                    for err in item["errors"]:
                        error_list.add(err)
                error_message = f"\n{model_output}\nThis XDL was not correct. These were the errors:\n{os.linesep.join(list(error_list))}\nPlease fix them."
                input_xdl = f"{prev_input_xdl} {error_message}"
        else:
            # If the <XDL> tags are missing
            error_message = f"\n{model_output}\nThis XDL was not correct. XDL should start with <XDL> and end with </XDL>. Please fix the errors."
            input_xdl = f"{prev_input_xdl} {error_message}"

    # Final result
    try:
        if correct_syntax:
            xdl = model_output
        else:
            xdl = "The correct XDL could not be generated."
    except Exception as e:
        print(f"Error: {e}")
        xdl = "Translation failed."

    if verbose:
        print(f"Final syntax validity: {correct_syntax}\n")
    return xdl
experiment_dir = "/home/shirish/Phd/xdl-generation/experiments"  
output_dir = os.path.join(experiment_dir, "outputs")
os.makedirs(output_dir, exist_ok=True)

# Limit to first 5 .txt files
experiment_files = sorted([
    f for f in os.listdir(experiment_dir)
    if f.endswith(".txt") and os.path.isfile(os.path.join(experiment_dir, f))
])[:5]

for idx, file_name in enumerate(experiment_files, start=1):
    file_path = os.path.join(experiment_dir, file_name)
    with open(file_path, "r") as f:
        input_xdl = f.read().strip()

    print(f"\nTranslating Experiment {idx}: {file_name}")
    translated_xdl = translate(input_xdl, verbose=True)

    output_file = os.path.join(output_dir, f"experiment_{idx}_output.txt")
    with open(output_file, "w") as out_f:
        out_f.write(translated_xdl)

    print(f"Saved translated XDL to: {output_file}")
