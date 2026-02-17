from huggingface_hub import create_repo, upload_folder

repo_name = "roberta-sentiment-classweighted"

create_repo(repo_name, private=False)

upload_folder(
    folder_path="roberta_sentiment_weighted",
    repo_id="parvj-06/roberta-sentiment-classweighted",
    commit_message="Upload class-weighted RoBERTa sentiment model"
)

print("Upload complete")