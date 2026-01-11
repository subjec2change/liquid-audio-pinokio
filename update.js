module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        message: "git pull"
      }
    },
    {
      method: "notify",
      params: {
        html: "Update complete! The repository has been updated to the latest version."
      }
    }
  ]
}
