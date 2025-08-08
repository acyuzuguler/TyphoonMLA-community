import os
import matplotlib.pyplot as plt

def plot_main_exp():
    plt.figure()
    plt.rcParams.update({'font.size': 18})  # Adjust the value as needed


    def parse_throughput(kernel, model, dataset, system_prompt, bsz):
        logname = 'logs/' + "_".join([str(v) for v in [kernel, model, dataset, system_prompt, bsz]]) + '.out'
        if not os.path.exists(logname):
            print(f"Log at {logname} could not be found. Skipping...")
            return None
        with open(logname, 'r') as f:
            lines = f.readlines()
            last_line = lines[-1].strip()
            throughput = float(last_line.split(" ")[-2])
        return throughput


    kernels=["flashmla", "flashinfer", "treemla"]
    models=["deepseekv3", "kimik2"]
    datasets=["mmlu", "gsm8k", "simpleqa"]
    system_prompts=["promptA", "promptB", "promptC"]
    bsz=1024
    
    kernel_labels = {"flashmla": "FlashMLA", "flashinfer": "FlashInfer", "treemla": "TreeMLA"}
    colors = {"treemla": "tab:red", "flashinfer": "steelblue", "flashmla": "tab:gray"}

    results = {}
    for kernel in kernels:
        if kernel not in results:
            results[kernel] = {}

        for model in models:
            if model not in results[kernel]:
                results[kernel][model] = {}

            for dataset in datasets:
                if dataset not in results[kernel][model]:
                    results[kernel][model][dataset] = {}

                for system_prompt in system_prompts:
                    throughput = parse_throughput(kernel, model, dataset, system_prompt, bsz)

                    results[kernel][model][dataset][system_prompt] = throughput
                    print("Kernel: {}, Model: {}, Dataset: {}, System Prompt: {}, throughput: {} TOPS/s".format(kernel, model, dataset, system_prompt, throughput))
                        
    fig, axs = plt.subplots(len(system_prompts), len(models)*len(datasets), figsize=(30, 15))

    for i, system_prompt in enumerate(system_prompts):
        for j, model in enumerate(models):
            for k, dataset in enumerate(datasets):
                ax = axs[i, j*len(system_prompts) + k]
                for kernel in kernels:
                    if kernel in results:
                        throughput = results[kernel][model][dataset][system_prompt]
                        if throughput:
                            ax.grid()
                            ax.bar(kernel_labels[kernel], throughput, color=colors[kernel], label=kernel)

                if i == len(system_prompts) - 1:
                    ax.set_xticks(range(len(kernels)), [kernel_labels[kernel] for kernel in kernels], rotation=45, ha='right')
                else:
                    ax.set_xticks([])

                title = "DeepSeekv3" if model == "deepseekv3" else "Kimi-K2"
                title += ", "
                title += "MMLU" if dataset == "mmlu" else "SimpleQA" if dataset == "simpleqa" else "GSM8K"
                ax.set_title(title)

                if j==0 and k==0:
                    ax.set_ylabel('Throughput (TOPS/s)')

    plt.savefig("plots/main_exp.png")

if __name__=="__main__":
    plot_main_exp()