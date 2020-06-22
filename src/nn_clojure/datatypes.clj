(ns nn-clojure.datatypes
  "TODO create namespace doc"
  (:require
   [nn-clojure.util :refer :all]
   [clojure.spec.alpha :as s]
   [clojure.spec.gen.alpha :as gen]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;; Specs
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
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
