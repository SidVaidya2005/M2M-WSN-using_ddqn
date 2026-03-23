const trainingForm = document.getElementById("trainingForm");

trainingForm.addEventListener("submit", async (e) => {
  e.preventDefault();

  const runButton = document.getElementById("runBtn");
  const buttonText = document.getElementById("runBtnText");
  const loader = document.getElementById("loader");
  const statusMessage = document.getElementById("statusMessage");
  const resultImage = document.getElementById("resultImage");
  const imagePlaceholder = document.getElementById("imagePlaceholder");
  const metricsGrid = document.getElementById("metricsGrid");
  const gifContainer = document.getElementById("gifContainer");

  const payload = {
    episodes: parseInt(document.getElementById("episodes").value),
    nodes: parseInt(document.getElementById("nodes").value),
    lr: parseFloat(document.getElementById("lr").value),
    gamma: parseFloat(document.getElementById("gamma").value),
    batch_size: parseInt(document.getElementById("batch_size").value),
    death_threshold:
      parseFloat(document.getElementById("death_threshold").value) / 100.0,
    seed: parseInt(document.getElementById("seed").value),
  };

  runButton.disabled = true;
  buttonText.textContent = "Training...";
  loader.style.display = "block";
  statusMessage.style.display = "none";
  statusMessage.className = "status-message";
  gifContainer.style.display = "none";

  try {
    const response = await fetch("/run_training", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await response.json();

    if (response.ok && data.status === "success") {
      statusMessage.textContent = data.message;
      statusMessage.classList.add("status-success");
      statusMessage.style.display = "block";

      const timestamp = new Date().getTime();
      resultImage.src = `${data.image_url}?t=${timestamp}`;

      resultImage.onload = () => {
        imagePlaceholder.style.display = "none";
        resultImage.style.display = "block";
      };

      if (data.gif_url) {
        const resultGif = document.getElementById("resultGif");
        resultGif.src = `${data.gif_url}?t=${timestamp}`;
        gifContainer.style.display = "block";
      }

      if (data.results) {
        document.getElementById("valBestLifetime").textContent =
          data.results.best_lifetime;
        document.getElementById("valBestEpisode").textContent =
          data.results.best_episode;
        document.getElementById("valAvgLifetime").textContent =
          data.results.avg_lifetime_final_10.toFixed(1);
        metricsGrid.style.display = "grid";
      }
    } else {
      throw new Error(data.message || "Unknown server error");
    }
  } catch (err) {
    statusMessage.textContent = `Error: ${err.message}`;
    statusMessage.classList.add("status-error");
    statusMessage.style.display = "block";
  } finally {
    runButton.disabled = false;
    buttonText.textContent = "Start Training";
    loader.style.display = "none";
  }
});
