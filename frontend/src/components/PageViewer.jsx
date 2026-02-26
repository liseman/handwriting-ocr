import { useState, useRef, useEffect, useCallback } from 'react';

export default function PageViewer({ imageSrc, ocrResults = [], selectedResultId, onSelectResult }) {
  const containerRef = useRef(null);
  const imgRef = useRef(null);
  const [imgDimensions, setImgDimensions] = useState({ width: 0, height: 0, naturalWidth: 0, naturalHeight: 0 });
  const [loaded, setLoaded] = useState(false);

  const updateDimensions = useCallback(() => {
    if (!imgRef.current) return;
    const img = imgRef.current;
    setImgDimensions({
      width: img.clientWidth,
      height: img.clientHeight,
      naturalWidth: img.naturalWidth,
      naturalHeight: img.naturalHeight,
    });
  }, []);

  useEffect(() => {
    if (!loaded) return;
    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, [loaded, updateDimensions]);

  const handleImageLoad = () => {
    setLoaded(true);
    updateDimensions();
  };

  // Calculate scale factors for rendering bounding boxes
  const scaleX = imgDimensions.naturalWidth > 0 ? imgDimensions.width / imgDimensions.naturalWidth : 1;
  const scaleY = imgDimensions.naturalHeight > 0 ? imgDimensions.height / imgDimensions.naturalHeight : 1;

  return (
    <div ref={containerRef} className="relative inline-block w-full">
      {!loaded && (
        <div className="skeleton w-full" style={{ paddingBottom: '141.4%' }} />
      )}
      <img
        ref={imgRef}
        src={imageSrc}
        alt="Document page"
        loading="lazy"
        onLoad={handleImageLoad}
        className={`w-full h-auto block transition-opacity duration-300 ${loaded ? 'opacity-100' : 'opacity-0'}`}
      />
      {loaded &&
        ocrResults.map((result) => {
          const bbox = result.bounding_box || result.bbox;
          if (!bbox) return null;

          // bbox format: { x, y, width, height } or [x, y, width, height]
          const x = Array.isArray(bbox) ? bbox[0] : bbox.x;
          const y = Array.isArray(bbox) ? bbox[1] : bbox.y;
          const w = Array.isArray(bbox) ? bbox[2] : bbox.width;
          const h = Array.isArray(bbox) ? bbox[3] : bbox.height;

          const isSelected = result.id === selectedResultId;

          return (
            <div
              key={result.id}
              onClick={() => onSelectResult?.(result.id)}
              className={`absolute border-2 cursor-pointer transition-all duration-200 ${
                isSelected
                  ? 'border-primary-500 bg-primary-500/20 shadow-lg'
                  : 'border-transparent hover:border-primary-300 hover:bg-primary-300/10'
              }`}
              style={{
                left: `${x * scaleX}px`,
                top: `${y * scaleY}px`,
                width: `${w * scaleX}px`,
                height: `${h * scaleY}px`,
              }}
              title={result.text}
            />
          );
        })}
    </div>
  );
}
