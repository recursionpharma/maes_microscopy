from huggingface_mae import MAEModel

# model = MAEModel.from_pretrained("models/phenom_beta", filename="last.pickle", from_state_dict=True)
model = MAEModel.from_pretrained("models/phenom_beta", filename="last.pickle")

huggingface_modelpath = "recursionpharma/test-pb-model"
model.push_to_hub(huggingface_modelpath)
# model.save_pretrained(huggingface_modelpath, push_to_hub=True, repo_id=huggingface_modelpath)

localpath = "models/phenom_beta_huggingface"
model.save_pretrained(localpath)
model = MAEModel.from_pretrained(localpath)
