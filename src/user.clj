(ns user
  "Define functions for interactive use at the repl along with comments to
  facilitate understanding the codebase."
  (:require [clojure.spec.alpha :as s]
            [clojure.spec.gen.alpha :as gen]
            [nn-clojure.core :refer :all]
            [nn-clojure.datatypes :as dt]
            [nn-clojure.domain :as do]
            [nn-clojure.train :as tr]
            [nn-clojure.util :refer :all]))

(declare try-it!)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; Constants
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; example nn and ctx
;;; 2421 layout
;;;   2 input nodes,
;;;   4 nodes in first hidden layer,
;;;   2 nodes in second hidden layer,
;;;   1 node in output layer
(def xor-2421-nn
  ;; input layer not shown since these are modeled as the input values to the
  ;; algorithm rather than neurons
  [;; first hidden layer (has 4 neurons)
   ;; each neuron has one weight for each of the neurons in the preceeding layer (input layer)
   [#:nn-clojure.datatypes{:weights [0.7 0.3],
                           :bias    -0.3}
    #:nn-clojure.datatypes{:weights [-0.6 0.4],
                           :bias    0.4}
    #:nn-clojure.datatypes{:weights [0.5 -0.5],
                           :bias    0.5}
    #:nn-clojure.datatypes{:weights [-0.4 -0.6],
                           :bias    -0.6}]
   ;; second hidden layer (has 2 neurons)
   [#:nn-clojure.datatypes{:weights [0.35 -0.38 -0.41 0.44],
                           :bias    0.4}
    #:nn-clojure.datatypes{:weights [0.65 -0.62 -0.59 0.56],
                           :bias    0.6}]
   ;; output layer (has 1 neuron)
   [#:nn-clojure.datatypes{:weights [0.45 0.55],
                            :bias   -0.5}]])

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;; Configuration examples
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; using a standard network show how different configuration options affect
;;;;; how it learns
(def xor-2421-ctx
  (merge conf-quick
         {::dt/nn             xor-2421-nn
          :activation/fn-name :nn-clojure.datatypes/sigmoid}))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; Training Examples
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; simplified training example
(defn single-pass-2421
  "Example of how context is changed when going through one iteration of forward
  and back propagation."
  []
  (-> xor-2421-ctx
      (assoc
       ;; set inputs
       :activation/nn [[1 0]]
       ;; set desired outputs
       ::dt/goals     [1])
      ;; determine actual outputs
      do/forward-propagation
      ;; learn from difference between outputs and goals
      do/backward-propagation))

;;; creating a training ready ctx example
(defn setup-xor
  "This creates a ctx with configuration needed to train a neural network to
  learn the xor boolean logic gate."
  [seed]
  (binding [dt/*r* (java.util.Random. seed)] ;; create the random number generator
    (let [;; generate a new unlearned neural network
          nn            (dt/nn [2 4 1])
          ;; activation-fn is used to determine a specific neurons output given
          ;; it's aggregated inputs
          activation-fn ::dt/sigmoid ;; see nn-clojure/datatypes for other implemented functions

          ;; set learning rate variables
          ;; use learning rate of 1.5  and max-epochs   3000 for < .01 target error rate
          ;; use learning rate of 0.01 and max-epochs 100000 for < .00005 target error rate
          ;; *NOTE* due to lein running :pre and :post conditions even when asserts are disabled
          ;;   attempting to learn a low target error rate can be very slow (minutes)
          learning-rate       1.5
          max-epochs       3000
          target-err-rate     0.01
          ;; function used to check if learning is successful (input is ctx)
          eval-fn          (fn [ctx] (< (tr/max-err ctx) 0.1))

          ;; create the patterns that the neural network will learn
          ;; the patterns below are for XOR boolean logic
          num-data-points  2 ;; the number of data points generated from each pattern
          ratio            0.5 ;; the ratio of generated data reserved for testing
          pattern-00       {:train/inputs     [0 0] ;; inputs to the neural network
                            ;; set some variablity of input data if you want the
                            ;; netowrk to learn noisy input data
                            ;; a random number will be generated between zero and
                            ;; range-down which is then subtracted from the
                            ;; equivalent input. the number of items in train/range-down
                            ;; must match the number of items in train/inputs
                            :train/range-down [0.0 0.0]
                            ;; range-up is similar to range-down but is added to the
                            ;; input variable instead of subtracted
                            :train/range-up   [0.0 0.0]
                            ;; goal is the outputs the network should learn to
                            ;; produce for the inputs
                            :train/goal       [0]}
          pattern-01       {:train/inputs [0 1]
                            :train/range-down [0.0 0.0]
                            :train/range-up [0.0 0]
                            :train/goal [1]}
          pattern-10       {:train/inputs [1 0]
                            :train/range-down [0.0 0.0]
                            :train/range-up [0.0 0.0]
                            :train/goal [1]}
          pattern-11       {:train/inputs [1 1]
                            :train/range-down [0.0 0.0]
                            :train/range-up [0.0 0]
                            :train/goal [0]}
          ;; the batch size determines how many data points (generated from the
          ;; patterns above) are applied to the neural network before an
          ;; average error is calculated and applied as part of the neural
          ;; network "learning" that batch of data
          ;; a larger batch size will tend to smooth out the learning process
          ;; such that training on several similar data points in a row doesn't
          ;; skew the network and affect its ability to learn different
          ;; patterns later
          batch-size 4 ;; number of data points in each training batch


          ;; create ctx data structure from above variables
          ctx {::dt/nn             nn
               ::dt/learning-rate  learning-rate
               :activation/fn-name activation-fn
               ::dt/rnd            dt/*r*
               :train/eval-fn      eval-fn
               :train/target       target-err-rate
               :train/batch-size   batch-size
               :train/min-epochs   1 ;; train at least once before checking for errors
               :train/max-epochs   max-epochs}
          ]

      (-> ctx
          (tr/+training-data num-data-points
                             ratio
                             pattern-00
                             pattern-01
                             pattern-10
                             pattern-11)))))

;;; example of training a ctx to completion
(defn xor-train
  "Train the xor network and return a string representing the result."
  ([]
   (let [arbitrary-value 42]
     (xor-train arbitrary-value)))
  ([seed]
   (let [ctx (setup-xor (or seed (rand-int 1000)))]
     (-> ctx
         tr/train
         tr/evaluate-result))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; Repl exploratory functions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defn gen
  "Convenience function to generate an example of a given spec."
  [spec]
  ;; see nn-clojure.datatypes for possible specs to generate and the
  ;;   implementation of custom generators
  ;;
  ;; examples:
  ;;   create a neuron with:
  ;;     (gen :nn-clojure.datatypes/neuron)
  ;;   layer:
  ;;     (gen :nn-clojure.datatypes/layer)
  ;;   neural network:
  ;;      (gen :nn-clojure.datatypes/nn)
  ;;   pattern for a nn to learn:
  ;;     (gen :train/pattern)
  (first (gen/sample (s/gen spec) 1)))

(defn try-it!
  []
  (let [num-threads (int (max (/ logical-threads 2) 1))]
    ;; this direct implementation of neural networks doesn't facilitate itself
    ;; to multi-threaded programming (you would want to use a model based off
    ;; linear algebra for that).
    ;;
    ;; however it can be useful to train multiple networks simultaneously since
    ;; training of any particular network is not guaranteed to be successful
    (println "Using" num-threads "thread(s) for training...")
    (pmap xor-train (take num-threads (cycle [nil])))))
