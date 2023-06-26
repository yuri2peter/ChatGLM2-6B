from transformers import AutoTokenizer, AutoModel


# 获取 tokenizer 和 model
def getTokenizerAndModel(bits=4):
    modelPath = "THUDM/chatglm2-6b"
    print(f"Using model: {modelPath}(bits {bits})")
    tokenizer = AutoTokenizer.from_pretrained(
        modelPath, trust_remote_code=True, revision="v1.0"
    )
    model = (
        AutoModel.from_pretrained(modelPath, trust_remote_code=True, revision="v1.0")
        .quantize(bits)
        .cuda()
    )
    model.eval()
    return tokenizer, model


# tokenizer, model = getTokenizerAndModel()
