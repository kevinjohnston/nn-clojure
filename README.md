nn-clojure
==========

Project for me to explore clojure.spec, generators, and learn about neural networks.

## Uses

The code within shouldn't be used as a library due to faster neural network libraries existing in the jvm ecosystem. Instead this exists purely as a reference for myself.

## Running

### Good ways
```bash
docker login
docker pull kevinjohnston/nn-clojure:1.0
docker run -it kevinjohnston/nn-clojure:1.0
```
OR
```bash
lein uberjar
java -jar ./target/uberjar/nn-clojure-0.1.0-SNAPSHOT-standalone.jar
```

### Bad way
`lein run`
    This approach isn't recommended due to lein calling `:pre` and `:post` checks
    even if asserts are disabled in `project.clj`, ultimately running the project
    this way will take an order of magnitude longer.

## Usage

Once the application is running you'll be provided with a simple prompt to train a new network on a selected boolean logic (OR/AND/NAND/XOR) with adjustable accuracy. The training will continue until it succeeds or times out with a message indicating the result.

If exploring the code in a repl I recommend starting with the `user` namespace as it has comments and example configuration to act as a guide.


## License

```
MIT License

Copyright (c) 2020 Kevin Johnston

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
