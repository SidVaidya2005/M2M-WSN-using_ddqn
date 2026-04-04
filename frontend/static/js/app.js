const trainingForm = document.getElementById("trainingForm");
const resultImage = document.getElementById("resultImage");
const imagePlaceholder = document.getElementById("imagePlaceholder");

function showResultImageIfAvailable() {
  if (!resultImage || !imagePlaceholder || !resultImage.getAttribute("src")) {
    return;
  }

  const revealImage = () => {
    imagePlaceholder.style.display = "none";
    resultImage.style.display = "block";
  };

  if (resultImage.complete && resultImage.naturalWidth > 0) {
    revealImage();
  } else {
    resultImage.addEventListener("load", revealImage, { once: true });
  }
}

showResultImageIfAvailable();

trainingForm.addEventListener("submit", async (e) => {
  e.preventDefault();

  const runButton = document.getElementById("runBtn");
  const buttonText = document.getElementById("runBtnText");
  const loader = document.getElementById("loader");
  const statusMessage = document.getElementById("statusMessage");
  const metricsGrid = document.getElementById("metricsGrid");
  const gifContainer = document.getElementById("gifContainer");

  const payload = {
    episodes: parseInt(document.getElementById("episodes").value),
    nodes: parseInt(document.getElementById("nodes").value),
    model_type: document.getElementById("model_type").value,
    learning_rate: parseFloat(document.getElementById("lr").value),
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
    const response = await fetch("http://127.0.0.1:5001/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await response.json();

    if (response.ok && data.status === "success") {
      const meanReward = Number(data.mean_reward);
      const maxReward = Number(data.max_reward);
      const bestLifetime = Number(data.results?.best_lifetime ?? maxReward);
      const bestEpisode = Number(
        data.results?.best_episode ?? data.episodes ?? 0,
      );
      const avgLifetimeFinal10 = Number(
        data.results?.avg_lifetime_final_10 ?? meanReward,
      );
      statusMessage.textContent =
        data.message ||
        `Training completed successfully. Mean reward: ${Number.isFinite(meanReward) ? meanReward.toFixed(2) : "N/A"}`;
      statusMessage.classList.add("status-success");
      statusMessage.style.display = "block";

      if (data.image_url) {
        const timestamp = new Date().getTime();
        resultImage.src = `http://127.0.0.1:5001${data.image_url}?t=${timestamp}`;
        resultImage.onload = () => {
          imagePlaceholder.style.display = "none";
          resultImage.style.display = "block";
        };
      } else {
        showResultImageIfAvailable();
      }

      if (data.gif_url) {
        const resultGif = document.getElementById("resultGif");
        const timestamp = new Date().getTime();
        resultGif.src = `${data.gif_url}?t=${timestamp}`;
        gifContainer.style.display = "block";
      }

      document.getElementById("valBestLifetime").textContent = Number.isFinite(
        bestLifetime,
      )
        ? bestLifetime.toFixed(2)
        : "-";
      document.getElementById("valBestEpisode").textContent = Number.isFinite(
        bestEpisode,
      )
        ? String(Math.trunc(bestEpisode))
        : "-";
      document.getElementById("valAvgLifetime").textContent = Number.isFinite(
        avgLifetimeFinal10,
      )
        ? avgLifetimeFinal10.toFixed(2)
        : "-";
      metricsGrid.style.display = "grid";
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
