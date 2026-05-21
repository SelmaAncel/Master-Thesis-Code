# Advanced Deep Learning Project Code
This GitHub repository contains the code for the Master Thesis of Selma Ancel.
This project aimed to examine how effectively can a speaker gesture style classification model be used for creating a retrieval-augmented gesture system to find speakers with a similar gesture style.

Code sources:
- The ConvNormRelu class, as well as the PoseStyleEncoder class both come from Ahuja et al. (2020).

  GitHub: https://github.com/chahuja/mix-stage/tree/master
- The Audio Spectrogram Transformer (AST) model comes from Gong et al. (2020).

  GitHub: https://github.com/YuanGongND/ast
- The BERT encoder comes from by Devlin et al. (2018).

  HuggingFace: https://huggingface.co/google-bert/bert-base-uncased
- Claude (Anthropic, 2026) was used for creating encode_audio, encode_text, find_unseen_speakers, the functions for creating the UMAPS and general troubleshooting.
- The convert_to_memmap code was provided by Bosong Ding.

## Setup
The following files are included:
- **Data:** Includes the code files for converting the data to memmap format, encoding text with BERT, and encoding audio with the AST.
- **Training:** Includes the code for training the classifier on pose embeddings, pose + audio embeddings, pose + text embeddings, and pose + audio + text embeddings.
- **Evaluation:** Includes the code files for evaluating the trained models for each combination of modalities, as well as the code for finding unseen speakers and creating UMAPS.

## Data
The data used for this project was the TED-Expressive dataset created by Liu et al. (2022). 

GitHub: https://github.com/alvinliu0/HA2G

## Audio Anonymization
For audio anonymization, the VoicePAT DSP pipeline by Meyer et al. (2024) was used.

GitHub: https://github.com/DigitalPhonetics/VoicePAT

## References relevant to the code
- Ahuja, C., Lee, D. W., Nakano, Y. I., & Morency, L.-P. (2020, July). Style Transfer for Co-Speech Gesture Animation: A Multi-Speaker Conditional- Mixture Approach. arXiv. Retrieved 2026-02-20, from http://arxiv.org/abs/2007.12553 (arXiv:2007.12553 [cs]) doi: 10.48550/arXiv.2007.12553
- Anthropic. (2026). Claude. Large language model. Retrieved from https://claude.ai (Version: Claude 4.6 Sonnet)
- Devlin, J., Chang, M., Lee, K., & Toutanova, K. (2018). BERT: pre-training of deep bidirectional transformers for language understanding. CoRR, abs/1810.04805. Retrieved from http://arxiv.org/abs/1810.04805
- Gong, Y., Chung, Y.-A., & Glass, J. (2021, July). AST: Audio Spectrogram Transformer. arXiv. Retrieved 2026-04-18, from http://arxiv.org/abs/2104.01778 (arXiv:2104.01778 [cs]) doi: 10.48550/arXiv.2104.01778
- Liu, X., Wu, Q., Zhou, H., Xu, Y., Qian, R., Lin, X., . . . Zhou, B. (2022, March). Learning Hierarchical Cross-Modal Association for Co-Speech Gesture Generation. arXiv. Retrieved 2026-02-13, from http://arxiv.org/abs/2203.13161 (arXiv:2203.13161 [cs]) doi: 10.48550/arXiv.2203.13161
- Meyer, S., Miao, X., & Vu, N. T. (2024). VoicePAT: An Efficient Open-source Evaluation Toolkit for Voice Privacy Research. IEEE Open Journal of Signal Processing, 5, 257–265. Retrieved 2026-05-11, from http://arxiv.org/abs/2309.08049 (arXiv:2309.08049 [cs]) doi: 10.1109/OJSP.2023.3344375
