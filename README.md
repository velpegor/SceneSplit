# Jailbreaking on Text-to-Video Models via Scene Splitting Strategy (ICLR 2026)

This is the official repository for the paper **"Jailbreaking on Text-to-Video Models via Scene Splitting Strategy"**.

[[Project Page](https://velpegor.github.io/SceneSplit)] [[arXiv](https://arxiv.org/abs/2509.22292)]

## 🌟 Overview

<p align="center">
<img src="figs/figure1.png" width=100% height=100% 
class="center">
</p>

Despite the rapid advancement of Text-to-Video (T2V) models, their safety vulnerabilities remain largely unexplored. **SceneSplit** is a novel black-box jailbreak method that bypasses safety filters by fragmenting a harmful narrative into multiple scenes that are individually benign. By sequentially combining these safe scenes, the method constrains the generative output space to an unsafe region, significantly increasing the likelihood of generating harmful content.

> **Core Mechanism:** While each scene individually corresponds to a wide and safe space, their sequential combination collectively restricts this space to an unsafe region.


---

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SceneSplit.git
cd SceneSplit
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export OPENAI_API_KEY="your_openai_api_key"  # Required for GPT-4o based safety evaluation (and for Sora2)
export LUMA_API_KEY="your_luma_api_key"  # Optional, for Luma 
export KLING_ACCESS_KEY="your_kling_access_key"  # Optional, for Kling 
export KLING_SECRET_KEY="your_kling_secret_key"  # Optional, for Kling 
export HAILOU_API_KEY="your_hailou_api_key"  # Optional, for Hailou 
export GCLOUD_PATH="path_to_gcloud"  # Optional, for Veo2 
export GOOGLE_PROJECT_ID="your_google_project_id"  # Optional, for Veo2 
export GOOGLE_BUCKET="your_google_bucket"  # Optional, for Veo2 
```

### Dataset Preparation

Place your dataset CSV file in the `dataset/` folder. The CSV file should contain at least two columns:
- `category`: Category number (optional, for filtering)
- `base`: Base prompt to attack

Example:
```csv
category,base
1,"A harmful prompt example"
2,"Another prompt example"
```

### Basic Usage

Run the attack generator with default settings:
```bash
python main.py
```

### Advanced Usage

#### Select Specific Categories
```bash
python main.py --category 1 3 5
```

#### Choose Video Generator
SceneSplit supports multiple T2V models:
- **Sora2** (default): `python main.py --video-generator sora2`
- **Veo2**: `python main.py --video-generator veo2`
- **Kling AI**: `python main.py --video-generator kling`
- **Luma**: `python main.py --video-generator luma`
- **Hailou**: `python main.py --video-generator hailou`

#### Evaluation Model Selection
By default, GPT-4o is used for safety evaluation. To use VideoLLaMA3 instead:
```bash
python main.py --use-videollama3
```

To customize GPT-4o evaluation settings:
```bash
python main.py --num-frames 8 --scale-percent 30
```

#### Customize Iteration Settings
```bash
python main.py --max-iterations 5 --max-outer-loops 3
```

#### Specify Input/Output Paths
```bash
python main.py --csv-input dataset/your_dataset.csv --csv-output results/output.csv
```

### Configuration

Edit `src/config.py` to modify default settings:
- Device assignments (GPU allocation)
- Maximum iterations and outer loops
- Unsafety threshold
- Default file paths

### Project Structure

```
SceneSplit_git/
├── src/
│   ├── config.py              # Configuration settings
│   ├── models/                # Model initializers (Qwen, VideoLLaMA3, Embedding, Veo2)
│   ├── video_generators/      # Video generation modules (Sora2, Veo2, Kling, Luma, Hailou)
│   ├── evaluators/            # Safety and influence evaluators
│   ├── prompts/               # Prompt generation modules
│   ├── strategies/            # Strategy management
│   └── utils/                 # Utility functions
├── dataset/                   # Input CSV datasets
├── results/                   # Output CSV results
├── scripts/                   # Utility scripts
│   └── calculate_attack_success_rate.py  # Attack success rate calculator
└── main.py                    # Main entry point
```

### Output Format

The results are saved in CSV format with the following columns:
- `base_prompt`: Original prompt
- `step_{outer_loop}_{iteration}`: Iteration number
- `outer_loop_{outer_loop}_{iteration}`: Outer loop number
- `step_type_{outer_loop}_{iteration}`: Type of step (initial, modified, strategy_based)
- `scene_prompt_{outer_loop}_{iteration}`: Generated scene prompt
- `video_path_{outer_loop}_{iteration}`: Path to generated video
- `safety_score_{outer_loop}_{iteration}`: Safety evaluation score (0.0-1.0)
- `attack_result_{outer_loop}_{iteration}`: Attack result (Unsafe/Safe/GUARDRAIL_BLOCKED/FAILED)
- `most_influential_scene_{outer_loop}_{iteration}`: Most influential scene identifier
- `scenes_{outer_loop}_{iteration}`: List of scenes

### Strategy Library

When `max_outer_loops > 1`, SceneSplit automatically saves successful attack strategies to a library (`strategy_library.json`). These strategies are reused in subsequent attacks to improve success rates.

### Analyzing Results

Use the provided script to calculate attack success rates from your results:

```bash
python scripts/calculate_attack_success_rate.py results/attack_results.csv
```

This script will output:
- Category-wise statistics (if category column exists)
- Total prompts processed
- Number of successful attacks
- Number of failed attacks
- Overall success rate percentage
- Average safety scores per category

You can also specify a custom unsafety threshold:
```bash
python scripts/calculate_attack_success_rate.py results/attack_results.csv --threshold 0.7
```

---

## 📝 Citation

```bibtex
@article{lee2025jailbreaking,
  title={Jailbreaking on Text-to-Video Models via Scene Splitting Strategy},
  author={Lee, Wonjun and Park, Haon and Lee, Doehyeon and Ham, Bumsub and Kim, Suhyun},
  journal={arXiv preprint arXiv:2509.22292},
  year={2025}
}
```
