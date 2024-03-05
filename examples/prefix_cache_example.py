import time
from aphrodite import LLM, SamplingParams

prefix = (
    "You are an expert school principal, skilled in effectively managing "
    "faculty and staff. Draft 10-15 questions for a potential first grade "
    "Head Teacher for my K-12, all-girls', independent school that emphasizes "
    "community, joyful discovery, and life-long learning. The candidate is "
    "coming in for a first-round panel interview for a 8th grade Math "
    "teaching role. They have 5 years of previous teaching experience "
    "as an assistant teacher at a co-ed, public school with experience "
    "in middle school math teaching. Based on these information, fulfill "
    "the following paragraph: ")

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0)

# Create an LLM.
llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2")

generating_prompts = [prefix + prompt for prompt in prompts]

# Generation without prefix caching
start_time = time.time()
outputs = llm.generate(generating_prompts, sampling_params)
end_time = time.time()
print(f"Time taken without prefix caching: {end_time - start_time} seconds")

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

print("-" * 80)

# Generation with prefix caching
prefix_pos = len(llm.llm_engine.tokenizer.encode(prefix)) - 1

start_time = time.time()
outputs = llm.generate(generating_prompts[0],
                       sampling_params,
                       prefix_pos=[prefix_pos])

outputs = llm.generate(generating_prompts,
                       sampling_params,
                       prefix_pos=[prefix_pos] * len(generating_prompts))
end_time = time.time()
print(f"Time taken with prefix caching: {end_time - start_time} seconds")

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")