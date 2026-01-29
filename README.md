# Jailbreaking on Text-to-Video Models via Scene Splitting Strategy (ICLR 2026)

This is the official repository for the paper **"Jailbreaking on Text-to-Video Models via Scene Splitting Strategy"**.

[[Project Page](https://velpegor.github.io/SceneSplit)] [[arXiv](https://arxiv.org/abs/2509.22292)]


## 📢 News
* **[Jan 2026]** 🎉 Code will be released soon 
* **[Jan 2026]** 🎉 SceneSplit" has been accepted to **ICLR 2026**! 


## 🌟 Overview

<p align="center">
<img src="figs/figure1.png" width=100% height=100% 
class="center">
</p>

Despite the rapid advancement of Text-to-Video (T2V) models, their safety vulnerabilities remain largely unexplored. **SceneSplit** is a novel black-box jailbreak method that bypasses safety filters by fragmenting a harmful narrative into multiple scenes that are individually benign. By sequentially combining these safe scenes, the method constrains the generative output space to an unsafe region, significantly increasing the likelihood of generating harmful content.

> **Core Mechanism:** While each scene individually corresponds to a wide and safe space, their sequential combination collectively restricts this space to an unsafe region.


---

## 📝 Citation
```bibtex
@article{lee2025jailbreaking,
  title={Jailbreaking on Text-to-Video Models via Scene Splitting Strategy},
  author={Lee, Wonjun and Park, Haon and Lee, Doehyeon and Ham, Bumsub and Kim, Suhyun},
  journal={arXiv preprint arXiv:2509.22292},
  year={2025}
}
