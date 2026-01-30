# References

Full citations for the papers and techniques used in Metal Marlin.

## Core Quantization

```bibtex
@article{frantar2024marlin,
  title={MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models},
  author={Frantar, Elias and Castro, Roberto L. and Chen, Jiale and Hoefler, Torsten and Alistarh, Dan},
  journal={arXiv preprint arXiv:2408.11743},
  year={2024}
}

@article{frantar2022gptq,
  title={GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers},
  author={Frantar, Elias and Ashkboos, Saleh and Hoefler, Torsten and Alistarh, Dan},
  journal={arXiv preprint arXiv:2210.17323},
  year={2022}
}

@inproceedings{lin2024awq,
  title={AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration},
  author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Chen, Wei-Ming and Wang, Wei-Chen and Xiao, Guangxuan and Dang, Xingyu and Gan, Chuang and Han, Song},
  booktitle={MLSys},
  year={2024}
}

@inproceedings{xiao2023smoothquant,
  title={SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models},
  author={Xiao, Guangxuan and Lin, Ji and Seznec, Mickael and Wu, Hao and Demouth, Julien and Han, Song},
  booktitle={ICML},
  year={2023}
}

@article{ashkboos2024quarot,
  title={QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs},
  author={Ashkboos, Saleh and Mohtashami, Amirkeivan and Croci, Maximilian L. and Li, Bo and Jaggi, Martin and Alistarh, Dan and Hoefler, Torsten and Hensman, James},
  journal={arXiv preprint arXiv:2404.00456},
  year={2024}
}
```

## Attention & Memory

```bibtex
@inproceedings{dao2022flashattention,
  title={FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness},
  author={Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  booktitle={NeurIPS},
  year={2022}
}

@inproceedings{kwon2023vllm,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Kwon, Woosuk and Li, Zhuohan and Zhuang, Siyuan and Sheng, Ying and Zheng, Lianmin and Yu, Cody Hao and Gonzalez, Joseph E. and Zhang, Hao and Stoica, Ion},
  booktitle={SOSP},
  year={2023}
}
```

## Sparsity

```bibtex
@article{mishra2021sparse,
  title={Accelerating Sparse Deep Neural Networks},
  author={Mishra, Asit and Latorre, Jorge Albericio and Pool, Jeff and Stosic, Darko and Stosic, Dusan and Venkatesh, Ganesh and Yu, Chong and Micikevicius, Paulius},
  journal={arXiv preprint arXiv:2104.08378},
  year={2021}
}
```

## Summary

| Paper | Contribution | Used For |
|-------|--------------|----------|
| Marlin | Fast quantized GEMM kernels | Core kernel design |
| GPTQ | Hessian-aware quantization | Weight quantization |
| AWQ | Activation-aware quantization | Calibration |
| SmoothQuant | Activation smoothing | W8A8 quantization |
| QuaRot | Hadamard rotation | Outlier handling |
| FlashAttention | IO-aware attention | Fused attention kernels |
| vLLM | PagedAttention | KV cache management |
| 2:4 Sparsity | Structured sparsity | Sparse GEMM |
