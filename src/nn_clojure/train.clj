(ns nn-clojure.train
  "Functions used to train a neural network."
  (:require
   [clojure.spec.alpha :as s]
   [nn-clojure.datatypes :as dt]
   [nn-clojure.domain :refer :all]
   [nn-clojure.util :refer :all]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; Constants
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; configuration defaults
(def max-raw-data-num 10)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; Functions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; helper functions
(defn- shuffle
  "Overrides clojure.core/shuffle to allow using a specific java.util.Random."
  [^java.util.Collection coll rnd]
  (let [al (java.util.ArrayList. coll)]
    (java.util.Collections/shuffle al rnd)
    (clojure.lang.RT/vector (.toArray al))))

(defn- do-if
  [ctx pred fn]
  (if (pred ctx)
    (fn ctx)
    ctx))

(defn update-all
  "Updates every value in a map with the given function."
  [m f]
  (zipmap (->> m keys)
          (->> m vals (map f))))

;;;;; train-point "getters"
(defn- train-point->example
  [train-point]
  (-> train-point :train/example))

(defn- train-point->input
  [train-point]
  (-> train-point :train/example first vector))

(defn- train-point->goals
  [train-point]
  (-> train-point :train/example second))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; Pattern functions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; helper functions
(declare pattern->data-set)
(defn- realize-patterns
  [size]
  (fn [patterns] (mapv #(pattern->data-set % size) patterns)))

;;;;; domain functions
(defn- pattern->data-point
  [{:train/keys [inputs range-up range-down] :as pattern}]
  {:pre  [(vex :train/pattern pattern)]
   :post [(vex :train/data-point %)]}
  (mapv (fn [input up down] (shift input (rand up) (rand down)))
        inputs
        range-up
        range-down))

(defn- pattern->data-set
  [{:train/keys [goal] :as pattern} size]
  {:pre  [(vex :train/pattern pattern)]
   :post [(vex :train/data-set %)]}
  (let [randomized-inputs (mapv pattern->data-point
                                (take size (repeat pattern)))]
    {pattern {:train/inputs randomized-inputs
              :train/goal   goal}}))

(defn- data-set->split-data
  [{:train/keys [inputs goal] :as data-set} ratio & [rnd]]
  {:pre  [(vex :train/data-set data-set) (vex :train/ratio ratio)]
   :post [(vex :train/split-data %)]}
  (let [examples            (reduce (fn [acc [k {:train/keys [inputs goal] :as v}]]
                                      (reduce conj acc (mapv (fn [input] [input goal k])
                                                             inputs)))
                                    []
                                    data-set)
        randomized-examples (shuffle examples (or rnd (java.util.Random.)))
        num-examples        (count examples)
        holdover-cutoff     (int (* ratio num-examples))
        training-examples   (subvec randomized-examples 0 holdover-cutoff)
        test-examples       (subvec randomized-examples holdover-cutoff)]
    {:train/examples training-examples
     :train/tests    test-examples}))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; Training point functions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defn- completed-batch?
  [train-point]
  (-> train-point :train/batch next nil?))

(defn- completed-epoch?
  [train-point]
  (and (-> train-point :train/epoch next nil?)
       (completed-batch? train-point)))

(defn- next-epoch
  [{:train/keys [epoch epochs batch batch-size examples]
    ::dt/keys   [rnd]
    :as ctx}]
  (let [epoch (partition-all batch-size (shuffle examples rnd))]
    {:train/epoch   epoch
     :train/batch   (-> epoch first)
     :train/example (-> epoch first first)
     :train/epochs  (inc (or epochs -1))}))

(defn- next-example
  [train-point]
  {:pre  [(vex :train/point train-point)]
   :post [(vex :train/point %)]}
  {:train/epoch   (-> train-point :train/epoch)
   :train/batch   (-> train-point :train/batch next)
   :train/example (-> train-point :train/batch next first)
   :train/epochs  (-> train-point :train/epochs)})

(defn- next-batch
  [train-point]
  {:pre  [(vex :train/point train-point)]
   :post [(vex :train/point %)]}
  {:train/epoch   (-> train-point :train/epoch next)
   :train/batch   (-> train-point :train/epoch next first)
   :train/example (-> train-point :train/epoch next first first)
   :train/epochs  (-> train-point :train/epochs)})

(defn- next-train-point
  [ctx train-point]
  (cond
    (completed-epoch? train-point) (-> train-point (merge ctx) next-epoch)
    (completed-batch? train-point) (-> train-point next-batch)
    :default                       (-> train-point next-example)))

(defn- training-seq
  ([ctx] (training-seq ctx (next-epoch ctx)))
  ([ctx train-point]
   (lazy-seq (cons train-point
                   (training-seq ctx (next-train-point ctx train-point))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; Error functions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; helper functions
(defn max-err
  "Return the maximum average error across all patterns. If no error exist to be
  averaged positive infinity is returned."
  [{:train/keys [error] :as ctx}]
  (if (not-empty error)
    (apply max (->> error vals (map :train/avg)))
    ##Inf))

(defn avg-err
  "Return the average error across all patterns. If no error exist to be
  averaged positive infinity is returned."
  [{:train/keys [error] :as ctx}]
  (if (not-empty error)
    (avg (->> error vals (map :train/avg)))
    ##Inf))

;;;;; error functions
(defn- update-err
  [{:train/keys [max-raw-data] :as ctx}
   {data-avg   :train/avg
    data-min   :train/min
    data-max   :train/max
    data-raw   :train/raw
    data-count :train/count
    :as err}]
  (let [max-raw-data (or max-raw-data max-raw-data-num)
        new-err      (total-error ctx)
        data-raw     (conj (or data-raw []) new-err)]
  {:train/avg   (mean data-raw)
   :train/min   (min (or data-min 1) new-err)
   :train/max   (max (or data-max 0) new-err)
   :train/std   (when (> (count data-raw) 1) (std-dev data-raw))
   :train/count (inc (or data-count (count data-raw)))
   :train/raw   (take max-raw-data-num data-raw)}))

(defn- record-error
  [{:train/keys [error max-raw-data] :as ctx}]
  (let [pattern (-> ctx :train/example last)]
    (update-in ctx [:train/error pattern] (partial update-err ctx))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; Step functions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defn- sum-adjustments
  [s1 s2]
  (cond
    (vector? s1) (mapv sum-adjustments s1 s2)
    (map? s1)    (merge-with sum-adjustments s1 s2)
    (number? s1) (+ s1 s2)))

(defn- avg-adjustments
  [s size]
  (cond
    (vector? s) (mapv #(avg-adjustments % size) s)
    (map? s)    (update-all s #(avg-adjustments % size))
    (number? s) (/ s size)))

(defn- +backprop-results
  [ctx]
  (update ctx :train/batch-adj conj (-> ctx :adjustments/nn vec)))

(defn- avg-batch
  [ctx]
  (let [summation  (reduce sum-adjustments (:train/batch-adj ctx))
        batch-size (count (:train/batch-adj ctx))
        average    (avg-adjustments summation batch-size)]
    (-> ctx
        (dissoc :train/batch-adj)
        (assoc ::dt/nn (vec average)))))

(defn- step
  [ctx]
  (-> ctx
      +backprop-results
      (do-if completed-batch? avg-batch)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; Clear functions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defn clear
  [ctx]
  (-> ctx
      ;; maybe change this to drop-keys
      (select-keys [::dt/nn
                    ;;:activation/nn
                    :activation/fn-name
                    ::dt/goals
                    ::dt/learning-rate
                    ;;::dt/rnd

                    :train/batch-size
                    :train/epoch
                    :train/epochs
                    :train/batch
                    :train/target
                    :train/error
                    :train/tests
                    :train/examples
                    :train/batch-adj

                    :train/annealing-fn
                    :train/eval-fn
                    :train/max-epochs
                    :train/min-epochs
                    :train/buffer
                    ])))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; Preparation functions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defn- prepare-example
  [ctx]
  (assoc ctx
         :train/example (train-point->example ctx)
         ::dt/goals     (train-point->goals   ctx)
         :activation/nn (train-point->input   ctx)))

(defn- anneal
  "Updates the learning rate via an annealing-fn if defined."
  [{:train/keys [annealing-fn] :as ctx}]
  (if annealing-fn
    (update ctx ::dt/learning-rate min (-> ctx avg-err annealing-fn))
    ctx))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; Testing functions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defn- prepare-test
  [ctx test]
  (assoc ctx
         :train/example test
         ::dt/goals     (-> test second)
         :activation/nn (-> test first vector)))

(declare test)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; Public api
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defn +training-data
  "Create training data and add to ctx."
  [{:keys [::dt/rnd] :as ctx} size ratio & patterns]
  (letfn [(patterns->data-sets [patterns]
            ((realize-patterns size) patterns))
          (reserve-test-data [data-sets]
            (map #(data-set->split-data % ratio rnd) data-sets))
          (combine-split-data [split-data]
            ;; needs to reduce and concat lists
            (reduce (fn [m1 m2]
                      (-> m1
                          (update :train/examples concat (:train/examples m2))
                          (update :train/tests concat (:train/tests m2))))
                    {}
                    split-data))
          (update-ctx [split-data]
            (merge ctx split-data))]
    (-> patterns
        patterns->data-sets
        reserve-test-data
        combine-split-data
        update-ctx)))

(defn +config
  "Setup ctx to include needed configuration data for training."
  [ctx batch-size target]
  (assoc ctx
         :train/batch-size batch-size
         :train/target target))

(defn train
  "Reducing function used to update ctx via an infinite sequence of training
  data."
  ([ctx] (reductions train ctx (training-seq ctx)))
  ([ctx train-point]
   (-> (merge ctx train-point)
       anneal
       prepare-example
       forward-propagation
       record-error
       backward-propagation
       step
       clear)))

(defn test
  "Tests a trained network to determine final success."
  [{:keys [:train/tests] :as ctx}]
  (reduce (fn [ctx test]
            (-> ctx
                (prepare-test test)
                forward-propagation
                record-error))
          (dissoc ctx :train/error)
          tests))
