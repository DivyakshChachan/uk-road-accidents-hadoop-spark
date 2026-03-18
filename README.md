# 🚦 Hadoop Cluster + PySpark Accident Severity Analysis

## 📌 Project Overview

This project demonstrates the end-to-end setup of a **Hadoop Distributed Cluster (Master-Slave Architecture)** and performing **distributed data analysis using PySpark on YARN**.

The dataset used is the **UK Road Accident Dataset**, where accident severity is analyzed using distributed processing.

---

## ⚙️ Technologies Used

* Hadoop (HDFS, YARN)
* PySpark
* Linux (Ubuntu)
* SSH (for cluster communication)
* Python
* HDFS (for distributed storage)

---

## 🏗️ System Architecture

* **Master Node**

  * NameNode
  * ResourceManager

* **Slave Node**

  * DataNode
  * NodeManager

Cluster is configured manually from scratch across **2 physical machines**.

---

## 🔧 Hadoop Configuration

The following configuration files were customized:

* `core-site.xml`
* `hdfs-site.xml`
* `yarn-site.xml`
* `mapred-site.xml`

Key configurations include:

* HDFS replication setup
* YARN resource allocation
* Namenode & Datanode communication
* Cluster resource scheduling

---

## 🚀 Cluster Setup Steps

1. Install Java & Hadoop on both machines
2. Configure SSH (passwordless login)
3. Set environment variables (`.bashrc`)
4. Configure Hadoop XML files
5. Format Namenode
6. Start Hadoop services:

   ```bash
   start-dfs.sh
   start-yarn.sh
   ```
7. Verify cluster:

   * `jps`
   * YARN Web UI

---

## 📂 HDFS Data Storage

Dataset was uploaded to HDFS using:

```bash
hdfs dfs -put accident_data.csv /input/
```

Features:

* Distributed storage across nodes
* Fault tolerance via replication
* Parallel data access

---

## 📊 PySpark Data Analysis

PySpark script: `pyspark/accident_analysis.py`

### Key Analysis Performed:

* Accident severity distribution
* Region-wise accident trends
* Time-based accident patterns
* Correlation between factors affecting severity

---

## ▶️ Running the PySpark Job on YARN

```bash
spark-submit \
--master yarn \
--deploy-mode cluster \
pyspark/accident_analysis.py
```

---

## 📈 Sample Insights

* Majority of accidents fall under **moderate severity**
* Certain regions show consistently higher accident rates
* Peak accident times observed during rush hours

---

## 📸 Screenshots

### YARN Resource Manager UI

<img width="1920" height="955" alt="Image" src="https://github.com/user-attachments/assets/7b3eab45-302a-4669-9cd4-98c7bd5a62c2" />

### Hadoop Cluster and Nodes

<img width="1920" height="955" alt="Image" src="https://github.com/user-attachments/assets/2ac43e96-87f5-49ba-a5f6-4a844091feb1" />

### Spark UI
<img width="1920" height="955" alt="Image" src="https://github.com/user-attachments/assets/386c26b9-87e2-4332-8ce7-9891d89292cc" />

---

## 📁 Repository Structure

```
configs/        → Hadoop configuration files  
scripts/        → Cluster setup & management scripts  
pyspark/        → Data analysis scripts  
data/           → Sample dataset  
screenshots/    → UI proof  
docs/           → Setup documentation  
```

---

## 💡 Key Learnings

* Hands-on experience with distributed systems
* Hadoop cluster setup from scratch
* HDFS data handling
* Running PySpark jobs on YARN
* Understanding resource allocation and job scheduling

---

## 🚀 Future Improvements

* Add more nodes to cluster (scalability testing)
* Integrate Hive for SQL-based analysis
* Use Spark MLlib for predictive modeling
* Real-time streaming with Spark Streaming

---

## 👨‍💻 Author
Divyaksh Chachan
Mohit Agrawal
B.Tech AI | SVNIT Surat

---
