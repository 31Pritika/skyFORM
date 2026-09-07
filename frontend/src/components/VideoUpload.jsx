import { useRef } from "react";
import { Upload } from "lucide-react";

function VideoUpload({
  onFileSelected,
  uploading,
  progress,
}) {
  const inputRef = useRef(null);

  const handleClick = () => {
    inputRef.current?.click();
  };

  const handleChange = (event) => {
    const file = event.target.files?.[0];

    if (file) {
      onFileSelected(file);
    }

    event.target.value = "";
  };

  return (
    <>
      <input
        ref={inputRef}
        type="file"
        accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska"
        onChange={handleChange}
        style={{ display: "none" }}
      />

      <button
        className="upload-button"
        onClick={handleClick}
        disabled={uploading}
      >
        <Upload size={17} />

        {uploading
          ? `Uploading ${progress}%`
          : "Upload drone footage"}
      </button>
    </>
  );
}

export default VideoUpload;