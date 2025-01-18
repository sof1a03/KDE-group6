import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { HttpParams } from '@angular/common/http';

export interface Book {
  name: string;
  url: string;
  bookid: string;
  genres: string[];
  publisher: string;
  year: number;
  ISBN: string;
  author: string;
}


@Injectable({
  providedIn: 'root'
})
export class BookService {
  private apiUrl = 'http://127.0.0.1:8000/api/search';
  private apiUrlRecs = 'http://127.0.0.1:8000/api/recommended_books';

  constructor(private http: HttpClient) { }

  getBooks(): Observable<Book[]> {
    return this.http.get<Book[]>(this.apiUrl);
  }

  getBook(id: string): Observable<Book> {
    return this.http.get<Book>(this.apiUrl);
  }

  searchBooks(
    isbn?: string,
    title?: string,
    author?: string,
    publisher?: string,
    categories?: string[],
    startYear?: string,
    endYear?: string,
    pageSize: number = 10,
    pageNum: number = 1
  ): Observable<Book[]> {
    console.log(this.http)
    let params = new HttpParams()
      .set('pageSize', pageSize.toString())
      .set('pageNum', pageNum.toString());

    if (isbn) {
      params = params.set('isbn', isbn);
    }
    if (title) {
      params = params.set('title', title);
    }
    if (author) {
      params = params.set('author', author);
    }
    if (publisher) {
      params = params.set('publisher', publisher);
    }
    if (categories) {
      categories.forEach(category => {
        params = params.append('categories', category);
      });
    }
    if (startYear) {
      params = params.set('start_year', startYear.toString());
    }
    if (endYear) {
      params = params.set('end_year', endYear.toString());
    }
    return this.http.get<Book[]>(this.apiUrl, { params });
  }

  getRecommendations(bookIds: string[]): Observable<Book[]> {
    let params = new HttpParams();
    bookIds.forEach(bookid=> {
      params = params.append('book_ids', bookid);
    });
    return this.http.get<Book[]>(this.apiUrlRecs, {

      params: params.set('top_n', '100')    });
  }}
