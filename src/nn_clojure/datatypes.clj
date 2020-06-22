(ns nn-clojure.datatypes
  "TODO create namespace doc"
  (:require
   [nn-clojure.util :refer :all]
   [clojure.spec.alpha :as s]
   [clojure.spec.gen.alpha :as gen]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; Specs
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; helpers
(defn- one? [n] (= n 1))
(defn- >=0 [n] (>= n 0))
(defn- valid-number-of-weights
  [nn]
  (if (one? (count nn))
    true
    (let [last-layer (nth nn (-> nn count dec))
          next-to-last-layer (nth nn (-> nn count dec dec))
          neuron (first last-layer)]
      (and (= (-> neuron ::weights count)
              (-> next-to-last-layer count))
           (valid-number-of-weights (butlast nn))))))

;;;;; specs
;; define numbers excluding ##NaN ##Inf ##-Inf
(s/def ::number (s/and number? #(Double/isFinite %)))
(s/def ::even (s/with-gen even?
                #(s/gen (s/and int? (fn [i] (= 0 (mod i 2)))))))
(s/def ::pos (s/with-gen pos?
               #(s/gen (s/and ::number pos?))))
(s/def ::not-neg (s/with-gen >=0
                   #(s/gen (s/and ::number >=0))))

(s/def ::weight ::number)
(s/def ::weights (s/and (s/coll-of ::weight :kind vector?) not-empty))
(s/def ::bias ::number)
(s/def ::activation ::number)
(s/def :activation/layer (s/coll-of ::activation :kind vector?))

(s/def ::input ::activation)
(s/def ::total-input ::number)
(s/def ::total-inputs (s/coll-of ::total-input :kind vector?))

(s/def ::goal ::activation)

(s/def ::delta ::number)
(s/def :delta/layer (s/coll-of ::delta :kind vector?))

(s/def ::layer (s/and (s/coll-of ::neuron :kind vector?)
                      not-empty
                      #(apply = (map (fn [neuron] (-> neuron ::weights count))
                                     %))))
