from hf2hw.main import export


def main():
    # model_id = "JackFram/llama-68m"
    # model_id = "google/gemma-3-270m-it"
    # model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    model_id = "google/gemma-3-270m-it"
    export(model_id, "ORT")


if __name__ == "__main__":
    main()
