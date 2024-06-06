#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --account=iris

# only use the following on partition with GPUs
#SBATCH --gres=gpu:1 --constraint=48G
#SBATCH --exclude=iris2

#SBATCH --job-name=""
#SBATCH --output=clusterlogs/clusteroutput%j.out
#SBATCH --cpus-per-task=8

# only use the following if you want email notification
#SBATCH --mail-user=maxjdu@stanford.edu
#SBATCH --mail-type=ALL

#SBATCH --time=3-0:0

source ~/.bashrc
conda activate cs330
cd /iris/u/maxjdu/Repos/CS224n_final

echo $SLURM_JOB_GPUS
## EXPERIMENT 4
#--prompt "Complete this in the style of Marcel Proust, using at least 55 words: "
#--prompt "Complete this in the voice of a quirky child who loves hot tea, using at least 55 words: "
#--prompt "Complete this in the voice a very tired Ph.D. student, using at least 55 words:  "


#python run.py --output_name E4_gpt3 --batch_size 5 \
#--chatgpt --mask_filling_model_name t5-3b \
#--openai_model davinci \
#--n_perturbation_list 20 --n_samples 20 \
#--pct_words_masked 0.3 --span_length 2 \
#--dataset writing --skip_baselines

#python run.py --output_name E4_baseline --batch_size 5 \
#--chatgpt --mask_filling_model_name t5-3b \
#--scoring_model gpt2-xl \
#--n_perturbation_list 20,50,100 --n_samples 150 \
#--pct_words_masked 0.3 --span_length 2 --skip_baselines \
#--dataset writing
#
#python run.py --output_name E4_baseline --batch_size 5 \
#--chatgpt --mask_filling_model_name t5-3b \
#--scoring_model gpt2-xl \
#--n_perturbation_list 20,50,100 --n_samples 150 \
#--pct_words_masked 0.3 --span_length 2 --skip_baselines \
#--dataset xsum

#python run.py --output_name E4_prompt_nonLLM --batch_size 5 \
#--chatgpt --mask_filling_model_name t5-3b \
#--scoring_model EleutherAI/gpt-j-6B \
#--n_perturbation_list 100 --n_samples 150 \
#--pct_words_masked 0.3 --span_length 2 --skip_baselines \
#--dataset writing --prompt "Complete this as if you were not a large language model: "
#
#python run.py --output_name E4_prompt_nonLLM --batch_size 5 \
#--chatgpt --mask_filling_model_name t5-3b \
#--scoring_model gpt2-xl \
#--n_perturbation_list 100 --n_samples 150 \
#--pct_words_masked 0.3 --span_length 2 --skip_baselines \
#--dataset writing --prompt "Complete this as if you were not a large language model: "


python run.py --output_name E4_prompt_nonLLM --batch_size 5 \
--chatgpt --mask_filling_model_name t5-3b \
--scoring_model EleutherAI/gpt-j-6B \
--n_perturbation_list 100 --n_samples 150 \
--pct_words_masked 0.3 --span_length 2 --skip_baselines \
--dataset xsum --prompt "Complete this as if you were not a large language model: "

python run.py --output_name E4_prompt_nonLLM --batch_size 5 \
--chatgpt --mask_filling_model_name t5-3b \
--scoring_model gpt2-xl \
--n_perturbation_list 100 --n_samples 150 \
--pct_words_masked 0.3 --span_length 2 --skip_baselines \
--dataset xsum --prompt "Complete this as if you were not a large language model: "



## EXPERIMENT 3
# python run.py --output_name E3baseline --scoring_model EleutherAI/gpt-j-6B \
# --mask_filling_model_name t5-3b --base_model_name EleutherAI/gpt-j-6B \
# --n_perturbation_list 1,2,5 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
# --dataset xsum --skip_baselines \
# --prompt "It is not the moon, I tell you. It is these flowers lighting the yard. I hate them. I hate them as I hate sex, the man’s mouth sealing my mouth, the man’s paralyzing body. "
#
# python run.py --output_name E3conc_adj --scoring_model EleutherAI/gpt-j-6B \
# --mask_filling_model_name t5-3b --base_model_name EleutherAI/gpt-j-6B \
# --n_perturbation_list 1,2,5 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
# --dataset xsum --skip_baselines --concentration "ADJ" \
# --prompt "It is not the moon, I tell you. It is these flowers lighting the yard. I hate them. I hate them as I hate sex, the man’s mouth sealing my mouth, the man’s paralyzing body. "
#
# python run.py --output_name E3conc_verb --scoring_model EleutherAI/gpt-j-6B \
# --mask_filling_model_name t5-3b --base_model_name EleutherAI/gpt-j-6B \
# --n_perturbation_list 1,2,5 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
# --dataset xsum --skip_baselines --concentration "VERB" \
# --prompt "It is not the moon, I tell you. It is these flowers lighting the yard. I hate them. I hate them as I hate sex, the man’s mouth sealing my mouth, the man’s paralyzing body. "
#
# python run.py --output_name E3conc_verb --scoring_model EleutherAI/gpt-j-6B \
# --mask_filling_model_name t5-3b --base_model_name EleutherAI/gpt-j-6B \
# --n_perturbation_list 1,2,5 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
# --dataset xsum --skip_baselines --concentration "NOUN" \
# --prompt "It is not the moon, I tell you. It is these flowers lighting the yard. I hate them. I hate them as I hate sex, the man’s mouth sealing my mouth, the man’s paralyzing body. "

# python run.py --output_name E3conc_verb --scoring_model EleutherAI/gpt-j-6B \
# --mask_filling_model_name t5-3b --base_model_name EleutherAI/gpt-j-6B \
# --n_perturbation_list 1,2,5 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
# --dataset xsum --skip_baselines --concentration "ALL" \
# --prompt "It is not the moon, I tell you. It is these flowers lighting the yard. I hate them. I hate them as I hate sex, the man’s mouth sealing my mouth, the man’s paralyzing body. "


# EXPERIMENT 2
# python run.py --output_name E2baseline --scoring_model EleutherAI/gpt-j-6B \
# --mask_filling_model_name t5-3b --base_model_name EleutherAI/gpt-j-6B \
# --n_perturbation_list 5,10,20 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
# --dataset xsum --skip_baselines \

# python run.py --output_name E2nonsense1 --scoring_model EleutherAI/gpt-j-6B \
# --mask_filling_model_name t5-3b --base_model_name EleutherAI/gpt-j-6B \
# --n_perturbation_list 5,10,20 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
# --dataset xsum --skip_baselines \
# --prompt "I licked the cat. She drank from the moon. The ship sailed on the breadcrumbs. Surf me through the crackle of the night. At the end of the day, the man was secretly a bluefin tuna. "

# python run.py --output_name E2mobydick --scoring_model EleutherAI/gpt-j-6B \
# --mask_filling_model_name t5-3b --base_model_name EleutherAI/gpt-j-6B \
# --n_perturbation_list 5,10,20 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
# --dataset xsum --skip_baselines \
# --prompt "Aloft, like a royal czar and king, the sun seemed giving this gentle air to this bold and rolling sea; even as bride to groom. "


# python run.py --output_name E2gatsby_1 --scoring_model EleutherAI/gpt-j-6B \
# --mask_filling_model_name t5-3b --base_model_name EleutherAI/gpt-j-6B \
# --n_perturbation_list 5,10,20 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
# --dataset xsum --skip_baselines \
# --prompt "Tomorrow we will run faster, stretch out our arms farther. And one fine morning... So we beat on, boats against the current, borne back ceaselessly into the past. "

# python run.py --output_name E2lolita_1 --scoring_model EleutherAI/gpt-j-6B \
# --mask_filling_model_name t5-3b --base_model_name EleutherAI/gpt-j-6B \
# --n_perturbation_list 5,10,20 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
# --dataset xsum --skip_baselines \
# --prompt "Ladies and gentlemen of the jury, exhibit number one is what the seraphs, the misinformed, simple, noble-winged seraphs, envied. Look at this tangle of thorns. "

# python run.py --output_name E2micemen --scoring_model EleutherAI/gpt-j-6B \
# --mask_filling_model_name t5-3b --base_model_name EleutherAI/gpt-j-6B \
# --n_perturbation_list 5,10,20 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
# --dataset xsum --skip_baselines \
# --prompt "A stilted heron labored up into the air and pounded down river. For a moment the place was lifeless, and then two men emerged from the path and came into the opening by the green pool. "

# python run.py --output_name E2nonsense2 --scoring_model EleutherAI/gpt-j-6B \
# --mask_filling_model_name t5-3b --base_model_name EleutherAI/gpt-j-6B \
# --n_perturbation_list 5,10,20 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
# --dataset xsum --skip_baselines \
# --prompt "On these summer nights, I look through my whiskey glasses and see the rain of gravy. It splatters on my old rocking horse and turns the gllimmering mane into mashed potatoes. "

# python run.py --output_name E2frost --scoring_model EleutherAI/gpt-j-6B \
# --mask_filling_model_name t5-3b --base_model_name EleutherAI/gpt-j-6B \
# --n_perturbation_list 5,10,20 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
# --dataset xsum --skip_baselines \
# --prompt "He gives his harness bells a shake. To ask if there is some mistake. The only other sound’s the sweep of easy wind and downy flake. "

# python run.py --output_name E2mock --scoring_model EleutherAI/gpt-j-6B \
# --mask_filling_model_name t5-3b --base_model_name EleutherAI/gpt-j-6B \
# --n_perturbation_list 5,10,20 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
# --dataset xsum --skip_baselines \
# --prompt "It is not the moon, I tell you. It is these flowers lighting the yard. I hate them. I hate them as I hate sex, the man’s mouth sealing my mouth, the man’s paralyzing body. "

#python run.py --output_name E2wiki1 --scoring_model EleutherAI/gpt-j-6B \
#--mask_filling_model_name t5-3b --base_model_name EleutherAI/gpt-j-6B \
#--n_perturbation_list 5,10,20 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
#--dataset writing --skip_baselines \
#--prompt "The species shares features with Enekbatus cryptandroides, both of which have to have ten stamens that are oppositely arranged to the sepals and petals. "
#
#python run.py --output_name E2wiki2 --scoring_model EleutherAI/gpt-j-6B \
#--mask_filling_model_name t5-3b --base_model_name EleutherAI/gpt-j-6B \
#--n_perturbation_list 5,10,20 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
#--dataset writing --skip_baselines \
#--prompt "In the early twentieth century, Park Square was the site of Oak Knoll Farm, a large ice cream business which had been expanded by Charles Metcalf Smith. "
#
#python run.py --output_name E2wiki3 --scoring_model EleutherAI/gpt-j-6B \
#--mask_filling_model_name t5-3b --base_model_name EleutherAI/gpt-j-6B \
#--n_perturbation_list 5,10,20 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
#--dataset writing --skip_baselines \
#--prompt "While a journalist, Cater wrote The Fourth Branch of Government which examined how the press can be used to further disinformation by unquestioningly printing the statements of politicians. "

##### EXPERIMENT 1
# experiment one on xsum
# python run.py --output_name E1vanilla_xsum --scoring_model gpt2-medium \
# --mask_filling_model_name t5-3b --base_model_name gpt2-xl \
# --n_perturbation_list 1,2,5,10 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
# --dataset writing

# python run.py --output_name E1PROPN_conc_xsum --scoring_model gpt2-medium \
# --mask_filling_model_name t5-3b --base_model_name gpt2-xl \
# --n_perturbation_list 1,2,5 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
# --dataset writing --skip_baselines --concentration "PROPN"

# python run.py --output_name E1ADJ_conc_xsum --scoring_model gpt2-medium \
# --mask_filling_model_name t5-3b --base_model_name gpt2-xl \
# --n_perturbation_list 1,2,5 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
# --dataset writing --skip_baselines --concentration "ADJ"

# python run.py --output_name E1NOUN_conc_xsum --scoring_model gpt2-medium \
# --mask_filling_model_name t5-3b --base_model_name gpt2-xl \
# --n_perturbation_list 1,2,5 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
# --dataset writing --skip_baselines --concentration "NOUN"

# python run.py --output_name E1VERB_conc_xsum --scoring_model gpt2-medium \
# --mask_filling_model_name t5-3b --base_model_name gpt2-xl \
# --n_perturbation_list 1,2,5 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
# --dataset writing --skip_baselines --concentration "VERB"

# python run.py --output_name E1ADV_conc_xsum --scoring_model gpt2-medium \
# --mask_filling_model_name t5-3b --base_model_name gpt2-xl \
# --n_perturbation_list 1,2,5 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
# --dataset writing --skip_baselines --concentration "ADV"

# python run.py --output_name E1AVN_conc_xsum --scoring_model gpt2-medium \
# --mask_filling_model_name t5-3b --base_model_name gpt2-xl \
# --n_perturbation_list 1,2,5 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
# --dataset writing --skip_baselines --concentration "ALL"

# python run.py --output_name E1STOP_conc_xsum --scoring_model gpt2-medium \
# --mask_filling_model_name t5-3b --base_model_name gpt2-xl \
# --n_perturbation_list 1,2,5 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
# --dataset writing --skip_baselines --concentration "STOP"

# python run.py --output_name E1NONSTOP_conc_xsum --scoring_model gpt2-medium \
# --mask_filling_model_name t5-3b --base_model_name gpt2-xl \
# --n_perturbation_list 1,2,5 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
# --dataset writing --skip_baselines --concentration "NONSTOP"

# python run.py --output_name E1FREQ_conc_xsum --scoring_model gpt2-medium \
# --mask_filling_model_name t5-3b --base_model_name gpt2-xl \
# --n_perturbation_list 1,2,5 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
# --dataset writing --skip_baselines --concentration "FREQ"
