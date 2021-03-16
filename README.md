
# Pytorch_learning


[![](https://img.shields.io/badge/version-1.0.0-brightgreen.svg)](https://github.com/bruce1408/Pytorch_learning)
![](https://img.shields.io/badge/platform-pytorch-brightgreen.svg)
![](https://img.shields.io/badge/python-3.7-blue.svg)


This repository provides tutorial code for deep learning researchers to learn [PyTorch](https://pytorch.org/)

PyTorch is a Python-based scientific computing package serving two broad purposes:

- A replacement for NumPy to use the power of GPUs and other accelerators.
- An automatic differentiation library that is useful to implement neural networks.


PyTorch, like most other deep learning libraries, supports reverse-mode automatic differentiation of
scalar functions (or vector-Jacobian products of functions with multiple outputs), the most important form 
of automatic differentiation for deep learning applications which usually differentiatea single scalar loss.
You write code as if you were executing tensor operations directly; however, 
instead of operating on Tensors (PyTorch’s equivalentof Numpy’s nd-arrays), 
the user manipulates Variables, which store extra metadata necessary for AD. Variables support a backward() method, 
which computes the gradient of all input Variables involved in computation of this quantity.


This repository contains:

1. [Computer Vision(CV)](/CV/cv.md) Implement complex computer vision algorithms.
2. [Natural language processing(NLP)](nlp.md) Examples to show how NLP can tacke real problem.Including the source code,
dataset, state-of-the art in NLP.
3. [Dataset](../data.md) All the data you can use in this project are under this directory.
4. [Reference](ref.md) Other reference materials for this project.


## Table of Contents

- [Project](#Project)
- [Install](#install)
- [Usage](#usage)
- [Example Readmes](#example-readmes)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)

## Project

#### Computer Vision 

- Image Classification 

    - AlexNet
    - VGGNet
    - ResNet
    - Inception-v1
    - Transfer-learning
    
- Generative adversarial network

- Open-Set domain adaption

#### Natural language processing

## Install

This project uses [node](http://nodejs.org) and [npm](https://npmjs.com). Go check them out if you don't have them locally installed.

```sh
$ npm install --global standard-readme-spec
```

## Usage

This is only a documentation package. You can print out [spec.md](spec.md) to your console:

```sh
$ standard-readme-spec
# Prints out the standard-readme spec
```

### Generator

To use the generator, look at [generator-standard-readme](https://github.com/RichardLitt/generator-standard-readme). There is a global executable to run the generator in that package, aliased as `standard-readme`.

## Badge

If your README is compliant with Standard-Readme and you're on GitHub, it would be great if you could add the badge. This allows people to link back to this Spec, and helps adoption of the README. The badge is **not required**.

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

To add in Markdown format, use this code:

```
[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)
```

## Example Readmes

To see how the specification has been applied, see the [example-readmes](example-readmes/).

## Related Efforts

- [Art of Readme](https://github.com/noffle/art-of-readme) - 💌 Learn the art of writing quality READMEs.
- [open-source-template](https://github.com/davidbgk/open-source-template/) - A README template to encourage open-source contributions.

## Maintainers

[@RichardLitt](https://github.com/RichardLitt).

## Contributing

Feel free to dive in! [Open an issue](https://github.com/RichardLitt/standard-readme/issues/new) or submit PRs.

Standard Readme follows the [Contributor Covenant](http://contributor-covenant.org/version/1/3/0/) Code of Conduct.

### Contributors

This project exists thanks to all the people who contribute. 
<a href="https://github.com/RichardLitt/standard-readme/graphs/contributors"><img src="https://opencollective.com/standard-readme/contributors.svg?width=890&button=false" /></a>


## License

[MIT](LICENSE) © Richard Littauer