# Speech Signal Improvement Challenge – ICASSP 2023

The Speech Signal Improvement Challenge Grand Challenge proposal at ICASSP 2023 is intended to stimulate research in the area of improving the speech signal quality in communication systems. The speech signal quality is measured with SIG in ITU-T P.804 and is still a top issue in audio communication and conferencing systems.

This challenge is to benchmark the performance of real-time speech enhancement models with a real (not simulated) test set. The audio scenario is the send signal in telecommunication; it does not include echo impairments. Participants will evaluate their speech enhancement model on a test set and submit the results (clips) for evaluation.

For more details about the challenge, please visit the challenge 
[website](https://www.microsoft.com/en-us/research/academic-program/speech-signal-improvement-challenge-icassp-2023/).
The paper will be released soon.

## Training data

The datasets are provided under the original terms that Microsoft received such datasets.
For the training data, we suggest participants to use [AEC-Challenge data](https://github.com/microsoft/AEC-Challenge) and [DNS-Challenge data](https://github.com/microsoft/DNS-Challenge), presented in the <b>Dataset licenses</b> section. 
Nevertheless, participants could use any other publicly available data for the training.

## Evaluation metrics
Our evaluation will be based on subjective listening test.
We suggest participants to evaluate models also in accordance with the [DNSMOS P.835](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS),
the <b>SIG</b> metric being directly correlated with the signal quality.
Nevertheless, participants could use any metrics for the model's evaluation.

## Datasets
* <b>Test set</b> is available in [test_data](https://github.com/microsoft/SIG-Challenge/tree/main/ICASSP2023/test_data) directory.
* <b>Blind set</b> is available in [blind_data](https://github.com/microsoft/SIG-Challenge/tree/main/ICASSP2023/blind_data) directory.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
