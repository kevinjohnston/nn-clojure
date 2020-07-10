FROM debian:buster

MAINTAINER Kevin

# Update and install dependencies from debian
RUN apt-get update  && \
  apt-get install -y git default-jdk wget

# download bitbucket repo
RUN git clone https://bitbucket.org/kljohnston/nn-clojure.git /root/nn-clojure

# Install lein (a build tool for clojure)
RUN wget https://raw.githubusercontent.com/technomancy/leiningen/stable/bin/lein \
    -O /bin/lein && chmod +x /bin/lein && /bin/lein

# Set java home, required to find jni c libraries for compilation
ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64/

# compile lein-native
WORKDIR /root/nn-clojure/
RUN lein uberjar

# Start in shell
ENTRYPOINT ["java", "-jar", "./target/uberjar/nn-clojure-0.1.0-SNAPSHOT-standalone.jar"]
