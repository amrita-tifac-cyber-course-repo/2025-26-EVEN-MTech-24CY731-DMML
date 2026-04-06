# DMML
## 📊 Dataset Information
This project is evaluated using real-world, structured production logs from the **Loghub** repository. While the primary analysis was conducted on Windows system events, the pipeline is architected to support multiple OS formats.

### Supported Datasets:
* **Windows 2k:** 2,000 structured logs from the Windows Component-Based Servicing (CBS) engine. [cite_start]Used to detect `HRESULT` failures and update errors.
* **Linux 2k:** 2,000 structured logs from Linux system history. Ideal for detecting SSH unauthorized access and sudo execution anomalies.

| OS Platform | Log Source | Download Link |
| :--- | :--- | :--- |
| **Windows** | System CBS | [Windows_2k.log_structured.csv](https://github.com/logpai/loghub/blob/master/Windows/Windows_2k.log_structured.csv) |
| **Linux** | Syslog | [Linux_2k.log_structured.csv](https://github.com/logpai/loghub/blob/master/Linux/Linux_2k.log_structured.csv) |

### Data Attributes:
The raw logs are pre-processed into a unified format containing:
1. **Timestamp:** For temporal context.
2. **Component:** To identify the originating system service.
3. **Level:** Severity labels (Info, Warning, Error).
4. **Content:** The raw event template for semantic vectorization.
