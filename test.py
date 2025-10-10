import modal

generate_next_token = modal.Function.from_name(
    "activation-vector-project", "generate_next_token"
)

result = generate_next_token.remote(text="The capital of France is")
print(result)
