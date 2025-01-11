import { Component, OnInit } from '@angular/core';
import { Input } from '@angular/core';
import { BookService } from '../../book-service';
import { Book } from '../../book-service';
import { SimpleChanges } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { CommonModule } from '@angular/common';
import { BookCardComponent } from "../book-card/book-card.component";

@Component({
  selector: 'app-book-page',
  imports: [BookCardComponent, CommonModule],
  standalone: true,
  templateUrl: './book-page.component.html',
  styleUrl: './book-page.component.css'
})

export class BookPageComponent implements OnInit {
  book = {} as Book;
  relatedBooks:Book[];

  constructor(
    private bookService: BookService,
    private route: ActivatedRoute
  ) {
    this.relatedBooks = [];
  }

  ngOnInit() {
    this.route.paramMap.subscribe(params => {
      const id = params.get('id');
      if (id) { // Check if id is not null
        this.bookService.getBook(id).subscribe(book => {
          this.book = book;
         });

        this.bookService.getMoreLike(id, 1, 5).subscribe(books => {

          this.relatedBooks = books;
        })
      } else {
        // Handle the case where 'id' is not found in the route
        console.error("Book ID not found in the route.");
        // You might want to redirect to an error page or display a message
      }
    });
  }}
