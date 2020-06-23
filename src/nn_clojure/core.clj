(ns nn-clojure.core
  "TODO namespace documentation"
  (:gen-class)
  (:require
   [clojure.spec.alpha :as s]
   [nn-clojure.datatypes :as dt]
   [nn-clojure.domain :as do]
   [nn-clojure.train :as tr]
   [nn-clojure.util :refer :all]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; Constants
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(def logical-threads (. (Runtime/getRuntime) availableProcessors))

(def conf-quick "Merge with a ctx when training a neural network for quick and dirty results."
  {::dt/learning-rate        1.5
   :train/max-epochs      4000
   :train/target             0.01})
(def conf-accurate "Merge with a ctx when training a neural network for accurate results."
  {::dt/learning-rate        0.01
   :train/max-epochs  10000000
   :train/target             0.0001})

(defn ctx
  "Return a new ctx including a new, randomly generated, neural network to run."
  ([] (ctx [2 4 1]))
  ([layers]
   (merge
    {::dt/nn             (dt/nn layers)
     ::dt/learning-rate  nil
     :activation/fn-name ::dt/sigmoid
     :train/max-epochs   nil
     :train/buffer       1.0
     :train/eval-fn      (fn [{:train/keys [buffer target] :as ctx} & [training]]
                           (< (* (if training
                                   buffer
                                   1)
                                 (tr/max-err ctx))
                              target))
     :train/batch-size   4
     ::dt/rnd            (java.util.Random.)
     :train/target       nil
     :train/max-raw-data 10}
    conf-quick)))

;;;;; training data
;;; xor
(def xor-examples  '([[0 0] [0] :00->0]
                     [[0 1] [1] :01->1]
                     [[1 0] [1] :10->1]
                     [[1 1] [0] :11->0]))
(def xor-tests     '([[0 0] [0] :00->0]
                     [[0 1] [1] :01->1]
                     [[1 0] [1] :10->1]
                     [[1 1] [0] :11->0]))
;;; and
(def and-examples  '([[0 0] [0] :00->0]
                     [[0 1] [0] :01->0]
                     [[1 0] [0] :10->0]
                     [[1 1] [1] :11->1]))
(def and-tests     '([[0 0] [0] :00->0]
                     [[0 1] [0] :01->0]
                     [[1 0] [0] :10->0]
                     [[1 1] [1] :11->1]))
;;; or
(def or-examples   '([[0 0] [0] :00->0]
                     [[0 1] [1] :01->1]
                     [[1 0] [1] :10->1]
                     [[1 1] [1] :11->1]))
(def or-tests      '([[0 0] [0] :00->0]
                     [[0 1] [1] :01->1]
                     [[1 0] [1] :10->1]
                     [[1 1] [1] :11->1]))
;;; nand
(def nand-examples '([[0 0] [1] :00->1]
                     [[0 1] [1] :01->1]
                     [[1 0] [1] :10->1]
                     [[1 1] [0] :11->0]))
(def nand-tests    '([[0 0] [1] :00->1]
                     [[0 1] [1] :01->1]
                     [[1 0] [1] :10->1]
                     [[1 1] [0] :11->0]))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; Functions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defn not-finished
  [{:train/keys [epochs target max-epochs eval-fn] :as ctx}]
  (if-let [epochs epochs]
    (not (or (eval-fn ctx true)
             (> epochs max-epochs)))
    true))

(defn result
  [{:train/keys [eval-fn] :as ctx}]
  (let [#_{time-str :str result :result} #_(with-out-str-data-map
                                         (time (first (drop-while
                                                       (not-finished ctx)
                                                       (tr/train ctx)))))
        time-str "nope"
        result (first (drop-while
                       not-finished
                       (tr/train ctx)))
        time-str (remove-formatting time-str)]
    (assoc result :time time-str)))

(defn evaluate-training
  "Use an evaluation function to determine if training was successful. Return a
  string indicating the result."
  [{:train/keys [eval-fn] :as ctx}]
  (let [ctx          (tr/test ctx)
        epochs       (-> ctx :train/epochs)]
    (str  "Training "
          (if (eval-fn ctx false)
            "SUCCEEDED"
            "FAILED")
          " after " epochs " epochs.")))

(defn -main
  [& args]
  (let [ctx  (merge (ctx) conf-quick)
        xor  (merge ctx
                    {:train/examples xor-examples
                     :train/tests    xor-tests})

        or   (merge ctx
                    {:train/examples or-examples
                     :train/tests    or-tests})

        and  (merge ctx
                    {:train/examples and-examples
                     :train/tests    and-tests})

        nand (merge ctx
                    {:train/examples nand-examples
                     :train/tests    nand-tests})]

    (and
     (pmap (fn [[ctx msg]]
             (->> ctx
                  (result)
                  (evaluate-training)
                  (println msg)))
           [[xor  "XOR: "]
            [or   "OR:  "]
            [and  "AND: "]
            [nand "NAND:"]])
     :finished)))
