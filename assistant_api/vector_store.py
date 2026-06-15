"""
Модуль работы с векторным хранилищем ChromaDB.
Обрабатывает загрузку документов, chunking и поиск по векторам.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
import os
import sys
from dotenv import load_dotenv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from proxy_config import create_openai_client


env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    # Пытаемся загрузить из текущей директории
    load_dotenv()


TEXT_EXTENSIONS = {'.txt', '.md'}


def list_data_files(data_dir: str) -> List[Path]:
    """Возвращает список текстовых файлов в папке data."""
    data_path = Path(data_dir)
    if not data_path.is_dir():
        return []
    return sorted(
        f for f in data_path.iterdir()
        if f.is_file() and f.suffix.lower() in TEXT_EXTENSIONS
    )


def prompt_for_data_loading(data_dir: str) -> bool:
    """
    Показывает список файлов в папке и спрашивает о загрузке.

    Returns:
        True — загрузить, False — пропустить (по умолчанию)
    """
    files = list_data_files(data_dir)
    if not files:
        print(f"⚠️  В папке '{data_dir}' нет текстовых файлов (.txt, .md)")
        return False

    print(f"\n📁 Файлы в папке '{data_dir}':")
    for i, file_path in enumerate(files, 1):
        size_kb = file_path.stat().st_size / 1024
        print(f"   {i}. {file_path.name} ({size_kb:.1f} KB)")

    answer = input("\nЗагрузить эти файлы в ChromaDB? (yes/no) [no]: ").strip().lower()
    if not answer:
        return False
    return answer in ('yes', 'y', 'да')


def prompt_for_collection_action(doc_count: int) -> str:
    """
    Спрашивает, как поступить с непустой коллекцией.

    Returns:
        'clear' — очистить и загрузить заново
        'add' — добавить к существующим документам
        'cancel' — отменить загрузку (по умолчанию)
    """
    print(f"\n📦 В коллекции уже есть документы ({doc_count} шт.).")
    answer = input(
        "Очистить коллекцию и загрузить заново или добавить файлы? "
        "(clear/add) [cancel]: "
    ).strip().lower()

    if not answer:
        return 'cancel'
    if answer in ('clear', 'c', 'очистить', 'заново', 'replace'):
        return 'clear'
    if answer in ('add', 'a', 'добавить', 'append'):
        return 'add'
    return 'cancel'


class VectorStore:
    """Векторное хранилище на основе ChromaDB."""
    
    def __init__(self, collection_name: str = "rag_collection", persist_directory: str = "./chroma_db"):
        """
        Инициализация векторного хранилища.
        
        Args:
            collection_name: имя коллекции в ChromaDB
            persist_directory: директория для хранения данных
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Инициализация ChromaDB клиента
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Получение или создание коллекции
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Коллекция '{collection_name}' загружена. Документов: {self.collection.count()}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Создана новая коллекция '{collection_name}'")
        
        # OpenAI клиент для создания embeddings (с поддержкой прокси)
        self.openai_client = create_openai_client()

    def clear_collection(self):
        """Удаляет коллекцию и создаёт пустую заново."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Коллекция '{self.collection_name}' очищена")
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """
        Умное разбиение текста на чанки с учётом семантики.
        
        Стратегия:
        1. Приоритет абзацам (разделение по \n\n)
        2. Разбиение длинных абзацев по предложениям
        3. Сохранение контекста через overlap
        4. Учёт минимального и максимального размера чанка
        
        Args:
            text: исходный текст
            chunk_size: целевой размер чанка в символах
            overlap: размер перекрытия между чанками
            
        Returns:
            список чанков
        """
        # Разделяем текст на абзацы
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Если абзац помещается в текущий чанк
            if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            
            # Если текущий чанк не пустой и добавление абзаца превысит размер
            elif current_chunk:
                chunks.append(current_chunk)
                # Добавляем overlap из конца предыдущего чанка
                overlap_text = self._get_overlap_text(current_chunk, overlap)
                current_chunk = overlap_text + "\n\n" + paragraph if overlap_text else paragraph
            
            # Если абзац слишком большой, разбиваем его на предложения
            else:
                if len(paragraph) > chunk_size:
                    # Разбиваем длинный абзац на предложения
                    sentence_chunks = self._split_long_paragraph(paragraph, chunk_size, overlap)
                    
                    # Добавляем все чанки кроме последнего
                    if sentence_chunks:
                        chunks.extend(sentence_chunks[:-1])
                        current_chunk = sentence_chunks[-1]
                else:
                    current_chunk = paragraph
        
        # Добавляем последний чанк
        if current_chunk:
            chunks.append(current_chunk)
        
        # Пост-обработка: фильтруем слишком короткие чанки
        chunks = [chunk for chunk in chunks if len(chunk) >= 50]
        
        return chunks
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """
        Получение текста для overlap из конца предыдущего чанка.
        Пытается взять целые предложения.
        
        Args:
            text: текст для извлечения overlap
            overlap_size: желаемый размер overlap
            
        Returns:
            текст overlap
        """
        if len(text) <= overlap_size:
            return text
        
        # Берём последние overlap_size символов
        overlap_candidate = text[-overlap_size:]
        
        # Ищем начало предложения в overlap
        sentence_starts = ['. ', '! ', '? ', '\n']
        best_start = 0
        
        for delimiter in sentence_starts:
            pos = overlap_candidate.find(delimiter)
            if pos != -1 and pos > best_start:
                best_start = pos + len(delimiter)
        
        if best_start > 0:
            return overlap_candidate[best_start:].strip()
        
        return overlap_candidate.strip()
    
    def _split_long_paragraph(self, paragraph: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Разбиение длинного абзаца на чанки по предложениям.
        
        Args:
            paragraph: абзац для разбиения
            chunk_size: целевой размер чанка
            overlap: размер перекрытия
            
        Returns:
            список чанков
        """
        # Разделяем на предложения
        import re
        sentences = re.split(r'([.!?]+\s+)', paragraph)
        
        # Собираем предложения обратно с их разделителями
        full_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                full_sentences.append(sentences[i] + sentences[i + 1])
            else:
                full_sentences.append(sentences[i])
        
        # Если осталось что-то в конце без разделителя
        if len(sentences) % 2 == 1:
            full_sentences.append(sentences[-1])
        
        chunks = []
        current_chunk = ""
        
        for sentence in full_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Если предложение помещается в текущий чанк
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                # Сохраняем текущий чанк
                if current_chunk:
                    chunks.append(current_chunk)
                    # Добавляем overlap
                    overlap_text = self._get_overlap_text(current_chunk, overlap)
                    current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                else:
                    # Если одно предложение больше chunk_size, всё равно добавляем его
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def load_documents(self, file_path: str):
        """
        Загрузка документов из одного файла в векторное хранилище.

        Args:
            file_path: путь к файлу с документами
        """
        if self.collection.count() > 0:
            print("Документы уже загружены в коллекцию")
            return

        file_name = Path(file_path).name
        print(f"\nОбработка файла: {file_name}")
        chunks = self._read_and_chunk_file(file_path)
        self._add_chunks(chunks, id_prefix=Path(file_path).stem)
        print(f"Загружено {len(chunks)} чанков из '{file_name}'")

    def load_documents_from_directory(self, data_dir: str, append: bool = False):
        """
        Загрузка всех текстовых файлов из папки в векторное хранилище.

        Args:
            data_dir: путь к папке с документами
            append: добавить к существующим документам (не заменять)
        """
        files = list_data_files(data_dir)
        if not files:
            raise FileNotFoundError(f"В папке '{data_dir}' нет файлов для загрузки")

        total_chunks = 0
        for file_path in files:
            print(f"\nОбработка файла: {file_path.name}")
            chunks = self._read_and_chunk_file(str(file_path))
            id_prefix = file_path.stem
            if append:
                id_prefix = f"{file_path.stem}_{self.collection.count()}"
            self._add_chunks(chunks, id_prefix=id_prefix, source=file_path.name)
            total_chunks += len(chunks)

        action = "добавлено" if append else "загружено"
        print(
            f"\n{action.capitalize()} {total_chunks} чанков из {len(files)} файлов "
            f"в коллекцию '{self.collection_name}'"
        )

    def _read_and_chunk_file(self, file_path: str) -> List[str]:
        """Чтение файла и разбиение на чанки."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл {file_path} не найден")

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        chunks = self._chunk_text(text)
        print(f"  Разбит на {len(chunks)} чанков")
        return chunks

    def _add_chunks(self, chunks: List[str], id_prefix: str, source: str = None):
        """Добавление чанков в ChromaDB с embeddings."""
        if not chunks:
            print("  Нет чанков для загрузки")
            return

        documents = []
        ids = []
        embeddings = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            embedding = self._create_embedding(chunk)
            documents.append(chunk)
            ids.append(f"{id_prefix}_{i}")
            embeddings.append(embedding)
            if source:
                metadatas.append({"source": source})

            if (i + 1) % 10 == 0:
                print(f"  Обработано {i + 1}/{len(chunks)} чанков")

        add_kwargs = {
            "documents": documents,
            "embeddings": embeddings,
            "ids": ids,
        }
        if metadatas:
            add_kwargs["metadatas"] = metadatas

        self.collection.add(**add_kwargs)
    
    def _create_embedding(self, text: str) -> List[float]:
        """
        Создание векторного представления текста через OpenAI.
        
        Args:
            text: текст для векторизации
            
        Returns:
            вектор embeddings
        """
        response = self.openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Поиск релевантных документов по запросу.
        
        Args:
            query: текст запроса
            top_k: количество документов для возврата
            
        Returns:
            список документов с метаданными
        """
        # Создание embedding для запроса
        query_embedding = self._create_embedding(query)
        
        # Поиск в ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Форматирование результатов
        documents = []
        if results['documents'] and len(results['documents'][0]):
            metadatas = results.get('metadatas', [[]])[0]
            for i in range(len(results['documents'][0])):
                doc = {
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                }
                if metadatas and i < len(metadatas) and metadatas[i]:
                    doc['source'] = metadatas[i].get('source')
                documents.append(doc)
        
        return documents
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Получение статистики коллекции.
        
        Returns:
            словарь со статистикой
        """
        return {
            'name': self.collection_name,
            'count': self.collection.count(),
            'persist_directory': self.persist_directory
        }


if __name__ == "__main__":
    # Тестирование векторного хранилища
    import sys
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Ошибка: установите переменную окружения OPENAI_API_KEY")
        sys.exit(1)
    
    vector_store = VectorStore(collection_name="test_collection")
    
    # Загрузка документов
    data_dir = "data"
    if list_data_files(data_dir):
        if prompt_for_data_loading(data_dir):
            vector_store.load_documents_from_directory(data_dir)
    
    # Поиск
    results = vector_store.search("Что такое машинное обучение?", top_k=3)
    print("\nРезультаты поиска:")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc['text'][:200]}...")
        print(f"   Distance: {doc['distance']}")
    
    # Статистика
    stats = vector_store.get_collection_stats()
    print(f"\nСтатистика: {stats}")

