import { useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getDocument, processDocument, processPage, getResults, submitCorrection } from '../api';
import { useToast } from '../hooks/useToast';
import PageViewer from '../components/PageViewer';
import OcrResultList from '../components/OcrResultList';

export default function DocumentView() {
  const { id } = useParams();
  const navigate = useNavigate();
  const toast = useToast();
  const queryClient = useQueryClient();
  const [selectedPageIndex, setSelectedPageIndex] = useState(0);
  const [selectedResultId, setSelectedResultId] = useState(null);

  const { data: doc, isLoading: docLoading } = useQuery({
    queryKey: ['document', id],
    queryFn: () => getDocument(id),
    enabled: !!id,
  });

  const pages = doc?.pages || [];
  const currentPage = pages[selectedPageIndex];
  const currentPageId = currentPage?.id;

  const { data: ocrResults, isLoading: ocrLoading } = useQuery({
    queryKey: ['ocrResults', currentPageId],
    queryFn: () => getResults(currentPageId),
    enabled: !!currentPageId,
  });

  const results = Array.isArray(ocrResults) ? ocrResults : ocrResults?.results || [];

  // Reset selected result when page changes
  useEffect(() => {
    setSelectedResultId(null);
  }, [selectedPageIndex]);

  const processDocMutation = useMutation({
    mutationFn: () => processDocument(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['document', id] });
      queryClient.invalidateQueries({ queryKey: ['ocrResults'] });
      toast.success('OCR processing started for all pages');
    },
    onError: () => {
      toast.error('Failed to start OCR processing.');
    },
  });

  const processPageMutation = useMutation({
    mutationFn: () => processPage(currentPageId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['ocrResults', currentPageId] });
      toast.success('Page processing started');
    },
    onError: () => {
      toast.error('Failed to process page.');
    },
  });

  const correctionMutation = useMutation({
    mutationFn: ({ resultId, text }) => submitCorrection(resultId, text),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['ocrResults', currentPageId] });
      toast.success('Correction saved');
    },
    onError: () => {
      toast.error('Failed to save correction.');
    },
  });

  const handleCorrect = (resultId, text) => {
    correctionMutation.mutate({ resultId, text });
  };

  const goToPrevPage = () => {
    if (selectedPageIndex > 0) setSelectedPageIndex(selectedPageIndex - 1);
  };

  const goToNextPage = () => {
    if (selectedPageIndex < pages.length - 1) setSelectedPageIndex(selectedPageIndex + 1);
  };

  if (docLoading) {
    return (
      <div className="max-w-6xl mx-auto px-4 py-6">
        <div className="skeleton h-8 w-48 mb-4" />
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <div className="skeleton w-full" style={{ paddingBottom: '141%' }} />
          </div>
          <div>
            <div className="skeleton h-6 w-32 mb-3" />
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="skeleton h-12 w-full mb-2" />
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (!doc) {
    return (
      <div className="max-w-6xl mx-auto px-4 py-12 text-center">
        <p className="text-gray-500">Document not found.</p>
        <button
          onClick={() => navigate('/documents')}
          className="mt-4 text-primary-600 hover:text-primary-700 text-sm font-medium"
        >
          Back to documents
        </button>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3 min-w-0">
          <button
            onClick={() => navigate('/documents')}
            className="p-1.5 text-gray-400 hover:text-gray-600 rounded-lg hover:bg-gray-100 transition-colors shrink-0"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
          </button>
          <div className="min-w-0">
            <h1 className="text-lg font-bold text-gray-900 truncate">
              {doc.name || doc.title || 'Untitled Document'}
            </h1>
            <p className="text-xs text-gray-400">
              {pages.length} page{pages.length !== 1 ? 's' : ''}
              {doc.created_at && <> &middot; {new Date(doc.created_at).toLocaleDateString()}</>}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <Link
            to={`/play?document=${id}`}
            className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium text-amber-700 bg-amber-50 rounded-lg hover:bg-amber-100 transition-colors"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Play Mode
          </Link>
          <button
            onClick={() => processDocMutation.mutate()}
            disabled={processDocMutation.isPending}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-primary-600 text-white text-sm font-medium rounded-lg hover:bg-primary-700 disabled:opacity-50 transition-colors"
          >
            {processDocMutation.isPending ? (
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
            ) : (
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            )}
            Process OCR
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Page thumbnails sidebar */}
        {pages.length > 1 && (
          <div className="lg:col-span-1">
            <div className="flex lg:flex-col gap-2 overflow-x-auto lg:overflow-y-auto lg:max-h-[calc(100vh-12rem)] pb-2 lg:pb-0 lg:pr-2">
              {pages.map((page, index) => (
                <button
                  key={page.id || index}
                  onClick={() => setSelectedPageIndex(index)}
                  className={`shrink-0 w-14 h-18 rounded-lg border-2 overflow-hidden transition-all ${
                    index === selectedPageIndex
                      ? 'border-primary-500 shadow-md'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  {page.thumbnail_url ? (
                    <img src={page.thumbnail_url} alt={`Page ${index + 1}`} className="w-full h-full object-cover" loading="lazy" />
                  ) : (
                    <div className="w-full h-full bg-gray-100 flex items-center justify-center text-xs text-gray-400">
                      {index + 1}
                    </div>
                  )}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Main page viewer */}
        <div className={pages.length > 1 ? 'lg:col-span-7' : 'lg:col-span-8'}>
          {currentPage ? (
            <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
              {/* Page navigation */}
              <div className="flex items-center justify-between px-4 py-2 bg-gray-50 border-b border-gray-200">
                <button
                  onClick={goToPrevPage}
                  disabled={selectedPageIndex === 0}
                  className="p-1 text-gray-500 hover:text-gray-700 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                  </svg>
                </button>
                <span className="text-sm text-gray-500">
                  Page {selectedPageIndex + 1} of {pages.length}
                </span>
                <button
                  onClick={goToNextPage}
                  disabled={selectedPageIndex === pages.length - 1}
                  className="p-1 text-gray-500 hover:text-gray-700 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </button>
              </div>

              {/* Image with overlays */}
              <div className="p-2">
                <PageViewer
                  imageSrc={currentPage.image_url || currentPage.url}
                  ocrResults={results}
                  selectedResultId={selectedResultId}
                  onSelectResult={setSelectedResultId}
                />
              </div>
            </div>
          ) : (
            <div className="bg-white rounded-xl border border-gray-200 p-12 text-center">
              <p className="text-gray-400">No pages in this document.</p>
            </div>
          )}
        </div>

        {/* OCR results panel */}
        <div className="lg:col-span-4">
          <div className="bg-white rounded-xl border border-gray-200 overflow-hidden sticky top-20">
            <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100">
              <h2 className="text-sm font-semibold text-gray-700">OCR Results</h2>
              {currentPageId && (
                <button
                  onClick={() => processPageMutation.mutate()}
                  disabled={processPageMutation.isPending}
                  className="text-xs text-primary-600 hover:text-primary-700 font-medium disabled:opacity-50"
                >
                  {processPageMutation.isPending ? 'Processing...' : 'Re-process'}
                </button>
              )}
            </div>

            <div className="max-h-[calc(100vh-16rem)] overflow-y-auto">
              {ocrLoading ? (
                <div className="p-4 space-y-3">
                  {[1, 2, 3, 4, 5].map((i) => (
                    <div key={i}>
                      <div className="skeleton h-4 w-full mb-1" />
                      <div className="skeleton h-3 w-1/3" />
                    </div>
                  ))}
                </div>
              ) : results.length > 0 ? (
                <OcrResultList
                  results={results}
                  selectedResultId={selectedResultId}
                  onSelectResult={setSelectedResultId}
                  onCorrect={handleCorrect}
                />
              ) : (
                <div className="p-6 text-center">
                  <svg className="w-10 h-10 text-gray-300 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  <p className="text-sm text-gray-400 mb-2">No OCR results for this page</p>
                  <button
                    onClick={() => processPageMutation.mutate()}
                    disabled={processPageMutation.isPending || !currentPageId}
                    className="text-sm text-primary-600 hover:text-primary-700 font-medium disabled:opacity-50"
                  >
                    {processPageMutation.isPending ? 'Processing...' : 'Process this page'}
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
