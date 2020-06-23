(ns nn-clojure.train
  "TODO create namespace doc"
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


